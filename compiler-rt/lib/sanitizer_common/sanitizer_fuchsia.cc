//===-- sanitizer_fuchsia.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and other sanitizer
// run-time libraries and implements Fuchsia-specific functions from
// sanitizer_common.h.
//===---------------------------------------------------------------------===//

#include "sanitizer_fuchsia.h"
#if SANITIZER_FUCHSIA

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_stacktrace.h"

#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <unwind.h>
#include <zircon/errors.h>
#include <zircon/process.h>
#include <zircon/syscalls.h>

namespace __sanitizer {

void NORETURN internal__exit(int exitcode) { _zx_process_exit(exitcode); }

uptr internal_sched_yield() {
  zx_status_t status = _zx_nanosleep(0);
  CHECK_EQ(status, ZX_OK);
  return 0;  // Why doesn't this return void?
}

static void internal_nanosleep(zx_time_t ns) {
  zx_status_t status = _zx_nanosleep(_zx_deadline_after(ns));
  CHECK_EQ(status, ZX_OK);
}

unsigned int internal_sleep(unsigned int seconds) {
  internal_nanosleep(ZX_SEC(seconds));
  return 0;
}

u64 NanoTime() { return _zx_time_get(ZX_CLOCK_UTC); }

uptr internal_getpid() {
  zx_info_handle_basic_t info;
  zx_status_t status =
      _zx_object_get_info(_zx_process_self(), ZX_INFO_HANDLE_BASIC, &info,
                          sizeof(info), NULL, NULL);
  CHECK_EQ(status, ZX_OK);
  uptr pid = static_cast<uptr>(info.koid);
  CHECK_EQ(pid, info.koid);
  return pid;
}

uptr GetThreadSelf() { return reinterpret_cast<uptr>(thrd_current()); }

uptr GetTid() { return GetThreadSelf(); }

void Abort() { abort(); }

int Atexit(void (*function)(void)) { return atexit(function); }

void SleepForSeconds(int seconds) { internal_sleep(seconds); }

void SleepForMillis(int millis) { internal_nanosleep(ZX_MSEC(millis)); }

void GetThreadStackTopAndBottom(bool, uptr *stack_top, uptr *stack_bottom) {
  pthread_attr_t attr;
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  void *base;
  size_t size;
  CHECK_EQ(pthread_attr_getstack(&attr, &base, &size), 0);
  CHECK_EQ(pthread_attr_destroy(&attr), 0);

  *stack_bottom = reinterpret_cast<uptr>(base);
  *stack_top = *stack_bottom + size;
}

void MaybeReexec() {}
void PrepareForSandboxing(__sanitizer_sandbox_arguments *args) {}
void DisableCoreDumperIfNecessary() {}
void InstallDeadlySignalHandlers(SignalHandlerType handler) {}
void StartReportDeadlySignal() {}
void ReportDeadlySignal(const SignalContext &sig, u32 tid,
                        UnwindSignalStackCallbackType unwind,
                        const void *unwind_context) {}
void SetAlternateSignalStack() {}
void UnsetAlternateSignalStack() {}
void InitTlsSize() {}

void PrintModuleMap() {}

bool SignalContext::IsStackOverflow() const { return false; }
void SignalContext::DumpAllRegisters(void *context) { UNIMPLEMENTED(); }
const char *SignalContext::Describe() const { UNIMPLEMENTED(); }

struct UnwindTraceArg {
  BufferedStackTrace *stack;
  u32 max_depth;
};

_Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx, void *param) {
  UnwindTraceArg *arg = static_cast<UnwindTraceArg *>(param);
  CHECK_LT(arg->stack->size, arg->max_depth);
  uptr pc = _Unwind_GetIP(ctx);
  if (pc < PAGE_SIZE) return _URC_NORMAL_STOP;
  arg->stack->trace_buffer[arg->stack->size++] = pc;
  return (arg->stack->size == arg->max_depth ? _URC_NORMAL_STOP
                                             : _URC_NO_REASON);
}

void BufferedStackTrace::SlowUnwindStack(uptr pc, u32 max_depth) {
  CHECK_GE(max_depth, 2);
  size = 0;
  UnwindTraceArg arg = {this, Min(max_depth + 1, kStackTraceMax)};
  _Unwind_Backtrace(Unwind_Trace, &arg);
  CHECK_GT(size, 0);
  // We need to pop a few frames so that pc is on top.
  uptr to_pop = LocatePcInTrace(pc);
  // trace_buffer[0] belongs to the current function so we always pop it,
  // unless there is only 1 frame in the stack trace (1 frame is always better
  // than 0!).
  PopStackFrames(Min(to_pop, static_cast<uptr>(1)));
  trace_buffer[0] = pc;
}

void BufferedStackTrace::SlowUnwindStackWithContext(uptr pc, void *context,
                                                    u32 max_depth) {
  CHECK_NE(context, nullptr);
  UNREACHABLE("signal context doesn't exist");
}

enum MutexState : int { MtxUnlocked = 0, MtxLocked = 1, MtxSleeping = 2 };

BlockingMutex::BlockingMutex() {
  // NOTE!  It's important that this use internal_memset, because plain
  // memset might be intercepted (e.g., actually be __asan_memset).
  // Defining this so the compiler initializes each field, e.g.:
  //   BlockingMutex::BlockingMutex() : BlockingMutex(LINKER_INITIALIZED) {}
  // might result in the compiler generating a call to memset, which would
  // have the same problem.
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  CHECK_EQ(owner_, 0);
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  if (atomic_exchange(m, MtxLocked, memory_order_acquire) == MtxUnlocked)
    return;
  while (atomic_exchange(m, MtxSleeping, memory_order_acquire) != MtxUnlocked) {
    zx_status_t status = _zx_futex_wait(reinterpret_cast<zx_futex_t *>(m),
                                        MtxSleeping, ZX_TIME_INFINITE);
    if (status != ZX_ERR_BAD_STATE)  // Normal race.
      CHECK_EQ(status, ZX_OK);
  }
}

void BlockingMutex::Unlock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  u32 v = atomic_exchange(m, MtxUnlocked, memory_order_release);
  CHECK_NE(v, MtxUnlocked);
  if (v == MtxSleeping) {
    zx_status_t status = _zx_futex_wake(reinterpret_cast<zx_futex_t *>(m), 1);
    CHECK_EQ(status, ZX_OK);
  }
}

void BlockingMutex::CheckLocked() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  CHECK_NE(MtxUnlocked, atomic_load(m, memory_order_relaxed));
}

uptr GetPageSize() { return PAGE_SIZE; }

uptr GetMmapGranularity() { return PAGE_SIZE; }

sanitizer_shadow_bounds_t ShadowBounds;

uptr GetMaxUserVirtualAddress() {
  ShadowBounds = __sanitizer_shadow_bounds();
  return ShadowBounds.memory_limit - 1;
}

uptr GetMaxVirtualAddress() {
  return GetMaxUserVirtualAddress();
}

static void *DoAnonymousMmapOrDie(uptr size, const char *mem_type,
                                  bool raw_report, bool die_for_nomem) {
  size = RoundUpTo(size, PAGE_SIZE);

  zx_handle_t vmo;
  zx_status_t status = _zx_vmo_create(size, 0, &vmo);
  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY || die_for_nomem)
      ReportMmapFailureAndDie(size, mem_type, "zx_vmo_create", status,
                              raw_report);
    return nullptr;
  }
  _zx_object_set_property(vmo, ZX_PROP_NAME, mem_type,
                          internal_strlen(mem_type));

  // TODO(mcgrathr): Maybe allocate a VMAR for all sanitizer heap and use that?
  uintptr_t addr;
  status = _zx_vmar_map(_zx_vmar_root_self(), 0, vmo, 0, size,
                        ZX_VM_FLAG_PERM_READ | ZX_VM_FLAG_PERM_WRITE, &addr);
  _zx_handle_close(vmo);

  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY || die_for_nomem)
      ReportMmapFailureAndDie(size, mem_type, "zx_vmar_map", status,
                              raw_report);
    return nullptr;
  }

  IncreaseTotalMmap(size);

  return reinterpret_cast<void *>(addr);
}

void *MmapOrDie(uptr size, const char *mem_type, bool raw_report) {
  return DoAnonymousMmapOrDie(size, mem_type, raw_report, true);
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  return MmapOrDie(size, mem_type);
}

void *MmapOrDieOnFatalError(uptr size, const char *mem_type) {
  return DoAnonymousMmapOrDie(size, mem_type, false, false);
}

uptr ReservedAddressRange::Init(uptr init_size, const char* name,
                                uptr fixed_addr) {
  base_ = MmapNoAccess(init_size);
  size_ = init_size;
  name_ = name;
  return reinterpret_cast<uptr>(base_);
}

// Uses fixed_addr for now.
// Will use offset instead once we've implemented this function for real.
uptr ReservedAddressRange::Map(uptr fixed_addr, uptr map_size) {
  return reinterpret_cast<uptr>(MmapFixedOrDieOnFatalError(fixed_addr,
                                                           map_size));
}

uptr ReservedAddressRange::MapOrDie(uptr fixed_addr, uptr map_size) {
  return reinterpret_cast<uptr>(MmapFixedOrDie(fixed_addr, map_size));
}

void ReservedAddressRange::Unmap(uptr addr, uptr size) {
  void* addr_as_void = reinterpret_cast<void*>(addr);
  uptr base_as_uptr = reinterpret_cast<uptr>(base_);
  // Only unmap at the beginning or end of the range.
  CHECK((addr_as_void == base_) || (addr + size == base_as_uptr + size_));
  CHECK_LE(size, size_);
  UnmapOrDie(reinterpret_cast<void*>(addr), size);
  if (addr_as_void == base_) {
    base_ = reinterpret_cast<void*>(addr + size);
  }
  size_ = size_ - size;
}

// MmapNoAccess and MmapFixedOrDie are used only by sanitizer_allocator.
// Instead of doing exactly what they say, we make MmapNoAccess actually
// just allocate a VMAR to reserve the address space.  Then MmapFixedOrDie
// uses that VMAR instead of the root.

zx_handle_t allocator_vmar = ZX_HANDLE_INVALID;
uintptr_t allocator_vmar_base;
size_t allocator_vmar_size;

void *MmapNoAccess(uptr size) {
  size = RoundUpTo(size, PAGE_SIZE);
  CHECK_EQ(allocator_vmar, ZX_HANDLE_INVALID);
  uintptr_t base;
  zx_status_t status =
      _zx_vmar_allocate(_zx_vmar_root_self(), 0, size,
                        ZX_VM_FLAG_CAN_MAP_READ | ZX_VM_FLAG_CAN_MAP_WRITE |
                            ZX_VM_FLAG_CAN_MAP_SPECIFIC,
                        &allocator_vmar, &base);
  if (status != ZX_OK)
    ReportMmapFailureAndDie(size, "sanitizer allocator address space",
                            "zx_vmar_allocate", status);

  allocator_vmar_base = base;
  allocator_vmar_size = size;
  return reinterpret_cast<void *>(base);
}

constexpr const char kAllocatorVmoName[] = "sanitizer_allocator";

static void *DoMmapFixedOrDie(uptr fixed_addr, uptr size, bool die_for_nomem) {
  size = RoundUpTo(size, PAGE_SIZE);

  zx_handle_t vmo;
  zx_status_t status = _zx_vmo_create(size, 0, &vmo);
  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY || die_for_nomem)
      ReportMmapFailureAndDie(size, kAllocatorVmoName, "zx_vmo_create", status);
    return nullptr;
  }
  _zx_object_set_property(vmo, ZX_PROP_NAME, kAllocatorVmoName,
                          sizeof(kAllocatorVmoName) - 1);

  DCHECK_GE(fixed_addr, allocator_vmar_base);
  uintptr_t offset = fixed_addr - allocator_vmar_base;
  DCHECK_LE(size, allocator_vmar_size);
  DCHECK_GE(allocator_vmar_size - offset, size);

  uintptr_t addr;
  status = _zx_vmar_map(
      allocator_vmar, offset, vmo, 0, size,
      ZX_VM_FLAG_PERM_READ | ZX_VM_FLAG_PERM_WRITE | ZX_VM_FLAG_SPECIFIC,
      &addr);
  _zx_handle_close(vmo);
  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY || die_for_nomem)
      ReportMmapFailureAndDie(size, kAllocatorVmoName, "zx_vmar_map", status);
    return nullptr;
  }

  IncreaseTotalMmap(size);

  return reinterpret_cast<void *>(addr);
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  return DoMmapFixedOrDie(fixed_addr, size, true);
}

void *MmapFixedOrDieOnFatalError(uptr fixed_addr, uptr size) {
  return DoMmapFixedOrDie(fixed_addr, size, false);
}

// This should never be called.
void *MmapFixedNoAccess(uptr fixed_addr, uptr size, const char *name) {
  UNIMPLEMENTED();
}

void *MmapAlignedOrDieOnFatalError(uptr size, uptr alignment,
                                   const char *mem_type) {
  CHECK_GE(size, PAGE_SIZE);
  CHECK(IsPowerOfTwo(size));
  CHECK(IsPowerOfTwo(alignment));

  zx_handle_t vmo;
  zx_status_t status = _zx_vmo_create(size, 0, &vmo);
  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY)
      ReportMmapFailureAndDie(size, mem_type, "zx_vmo_create", status, false);
    return nullptr;
  }
  _zx_object_set_property(vmo, ZX_PROP_NAME, mem_type,
                          internal_strlen(mem_type));

  // TODO(mcgrathr): Maybe allocate a VMAR for all sanitizer heap and use that?

  // Map a larger size to get a chunk of address space big enough that
  // it surely contains an aligned region of the requested size.  Then
  // overwrite the aligned middle portion with a mapping from the
  // beginning of the VMO, and unmap the excess before and after.
  size_t map_size = size + alignment;
  uintptr_t addr;
  status = _zx_vmar_map(_zx_vmar_root_self(), 0, vmo, 0, map_size,
                        ZX_VM_FLAG_PERM_READ | ZX_VM_FLAG_PERM_WRITE, &addr);
  if (status == ZX_OK) {
    uintptr_t map_addr = addr;
    uintptr_t map_end = map_addr + map_size;
    addr = RoundUpTo(map_addr, alignment);
    uintptr_t end = addr + size;
    if (addr != map_addr) {
      zx_info_vmar_t info;
      status = _zx_object_get_info(_zx_vmar_root_self(), ZX_INFO_VMAR, &info,
                                   sizeof(info), NULL, NULL);
      if (status == ZX_OK) {
        uintptr_t new_addr;
        status =
            _zx_vmar_map(_zx_vmar_root_self(), addr - info.base, vmo, 0, size,
                         ZX_VM_FLAG_PERM_READ | ZX_VM_FLAG_PERM_WRITE |
                             ZX_VM_FLAG_SPECIFIC_OVERWRITE,
                         &new_addr);
        if (status == ZX_OK) CHECK_EQ(new_addr, addr);
      }
    }
    if (status == ZX_OK && addr != map_addr)
      status = _zx_vmar_unmap(_zx_vmar_root_self(), map_addr, addr - map_addr);
    if (status == ZX_OK && end != map_end)
      status = _zx_vmar_unmap(_zx_vmar_root_self(), end, map_end - end);
  }
  _zx_handle_close(vmo);

  if (status != ZX_OK) {
    if (status != ZX_ERR_NO_MEMORY)
      ReportMmapFailureAndDie(size, mem_type, "zx_vmar_map", status, false);
    return nullptr;
  }

  IncreaseTotalMmap(size);

  return reinterpret_cast<void *>(addr);
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  size = RoundUpTo(size, PAGE_SIZE);

  zx_status_t status = _zx_vmar_unmap(_zx_vmar_root_self(),
                                      reinterpret_cast<uintptr_t>(addr), size);
  if (status != ZX_OK) {
    Report("ERROR: %s failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           SanitizerToolName, size, size, addr);
    CHECK("unable to unmap" && 0);
  }

  DecreaseTotalMmap(size);
}

// This is used on the shadow mapping, which cannot be changed.
// Zircon doesn't have anything like MADV_DONTNEED.
void ReleaseMemoryPagesToOS(uptr beg, uptr end) {}

void DumpProcessMap() {
  UNIMPLEMENTED();  // TODO(mcgrathr): write it
}

bool IsAccessibleMemoryRange(uptr beg, uptr size) {
  // TODO(mcgrathr): Figure out a better way.
  zx_handle_t vmo;
  zx_status_t status = _zx_vmo_create(size, 0, &vmo);
  if (status == ZX_OK) {
    while (size > 0) {
      size_t wrote;
      status = _zx_vmo_write(vmo, reinterpret_cast<const void *>(beg), 0, size,
                             &wrote);
      if (status != ZX_OK) break;
      CHECK_GT(wrote, 0);
      CHECK_LE(wrote, size);
      beg += wrote;
      size -= wrote;
    }
    _zx_handle_close(vmo);
  }
  return status == ZX_OK;
}

// FIXME implement on this platform.
void GetMemoryProfile(fill_profile_f cb, uptr *stats, uptr stats_size) {}

bool ReadFileToBuffer(const char *file_name, char **buff, uptr *buff_size,
                      uptr *read_len, uptr max_len, error_t *errno_p) {
  zx_handle_t vmo;
  zx_status_t status = __sanitizer_get_configuration(file_name, &vmo);
  if (status == ZX_OK) {
    uint64_t vmo_size;
    status = _zx_vmo_get_size(vmo, &vmo_size);
    if (status == ZX_OK) {
      if (vmo_size < max_len) max_len = vmo_size;
      size_t map_size = RoundUpTo(max_len, PAGE_SIZE);
      uintptr_t addr;
      status = _zx_vmar_map(_zx_vmar_root_self(), 0, vmo, 0, map_size,
                            ZX_VM_FLAG_PERM_READ, &addr);
      if (status == ZX_OK) {
        *buff = reinterpret_cast<char *>(addr);
        *buff_size = map_size;
        *read_len = max_len;
      }
    }
    _zx_handle_close(vmo);
  }
  if (status != ZX_OK && errno_p) *errno_p = status;
  return status == ZX_OK;
}

void RawWrite(const char *buffer) {
  __sanitizer_log_write(buffer, internal_strlen(buffer));
}

void CatastrophicErrorWrite(const char *buffer, uptr length) {
  __sanitizer_log_write(buffer, length);
}

char **StoredArgv;
char **StoredEnviron;

char **GetArgv() { return StoredArgv; }

const char *GetEnv(const char *name) {
  if (StoredEnviron) {
    uptr NameLen = internal_strlen(name);
    for (char **Env = StoredEnviron; *Env != 0; Env++) {
      if (internal_strncmp(*Env, name, NameLen) == 0 && (*Env)[NameLen] == '=')
        return (*Env) + NameLen + 1;
    }
  }
  return nullptr;
}

uptr ReadBinaryName(/*out*/ char *buf, uptr buf_len) {
  const char *argv0 = StoredArgv[0];
  if (!argv0) argv0 = "<UNKNOWN>";
  internal_strncpy(buf, argv0, buf_len);
  return internal_strlen(buf);
}

uptr ReadLongProcessName(/*out*/ char *buf, uptr buf_len) {
  return ReadBinaryName(buf, buf_len);
}

uptr MainThreadStackBase, MainThreadStackSize;

bool GetRandom(void *buffer, uptr length, bool blocking) {
  CHECK_LE(length, ZX_CPRNG_DRAW_MAX_LEN);
  size_t size;
  CHECK_EQ(_zx_cprng_draw(buffer, length, &size), ZX_OK);
  CHECK_EQ(size, length);
  return true;
}

}  // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
void __sanitizer_startup_hook(int argc, char **argv, char **envp,
                              void *stack_base, size_t stack_size) {
  __sanitizer::StoredArgv = argv;
  __sanitizer::StoredEnviron = envp;
  __sanitizer::MainThreadStackBase = reinterpret_cast<uintptr_t>(stack_base);
  __sanitizer::MainThreadStackSize = stack_size;
}

void __sanitizer_set_report_path(const char *path) {
  // Handle the initialization code in each sanitizer, but no other calls.
  // This setting is never consulted on Fuchsia.
  DCHECK_EQ(path, common_flags()->log_path);
}

void __sanitizer_set_report_fd(void *fd) {
  UNREACHABLE("not available on Fuchsia");
}
}  // extern "C"

#endif  // SANITIZER_FUCHSIA
