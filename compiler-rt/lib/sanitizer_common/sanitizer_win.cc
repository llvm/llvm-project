//===-- sanitizer_win.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements windows-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS

#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <dbghelp.h>
#include <io.h>
#include <psapi.h>
#include <stdlib.h>

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

#include "sanitizer_syscall_generic.inc"

// --------------------- sanitizer_common.h
uptr GetPageSize() {
  // FIXME: there is an API for getting the system page size (GetSystemInfo or
  // GetNativeSystemInfo), but if we use it here we get test failures elsewhere.
  return 1U << 14;
}

uptr GetMmapGranularity() {
  return 1U << 16;  // FIXME: is this configurable?
}

uptr GetMaxVirtualAddress() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (uptr)si.lpMaximumApplicationAddress;
}

bool FileExists(const char *filename) {
  UNIMPLEMENTED();
}

uptr internal_getpid() {
  return GetProcessId(GetCurrentProcess());
}

// In contrast to POSIX, on Windows GetCurrentThreadId()
// returns a system-unique identifier.
uptr GetTid() {
  return GetCurrentThreadId();
}

uptr GetThreadSelf() {
  return GetTid();
}

#if !SANITIZER_GO
void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  MEMORY_BASIC_INFORMATION mbi;
  CHECK_NE(VirtualQuery(&mbi /* on stack */, &mbi, sizeof(mbi)), 0);
  // FIXME: is it possible for the stack to not be a single allocation?
  // Are these values what ASan expects to get (reserved, not committed;
  // including stack guard page) ?
  *stack_top = (uptr)mbi.BaseAddress + mbi.RegionSize;
  *stack_bottom = (uptr)mbi.AllocationBase;
}
#endif  // #if !SANITIZER_GO

void *MmapOrDie(uptr size, const char *mem_type) {
  void *rv = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == 0) {
    Report("ERROR: %s failed to "
           "allocate 0x%zx (%zd) bytes of %s (error code: %d)\n",
           SanitizerToolName, size, size, mem_type, GetLastError());
    CHECK("unable to mmap" && 0);
  }
  return rv;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!size || !addr)
    return;

  if (VirtualFree(addr, size, MEM_DECOMMIT) == 0) {
    Report("ERROR: %s failed to "
           "deallocate 0x%zx (%zd) bytes at address %p (error code: %d)\n",
           SanitizerToolName, size, size, addr, GetLastError());
    CHECK("unable to unmap" && 0);
  }
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size, const char *name) {
  // FIXME: is this really "NoReserve"? On Win32 this does not matter much,
  // but on Win64 it does.
  (void)name; // unsupported
  void *p = VirtualAlloc((LPVOID)fixed_addr, size,
      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (p == 0)
    Report("ERROR: %s failed to "
           "allocate %p (%zd) bytes at %p (error code: %d)\n",
           SanitizerToolName, size, size, fixed_addr, GetLastError());
  return p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  return MmapFixedNoReserve(fixed_addr, size);
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  // FIXME: make this really NoReserve?
  return MmapOrDie(size, mem_type);
}

void *MmapNoAccess(uptr fixed_addr, uptr size, const char *name) {
  (void)name; // unsupported
  void *res = VirtualAlloc((LPVOID)fixed_addr, size,
                           MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
  if (res == 0)
    Report("WARNING: %s failed to "
           "mprotect %p (%zd) bytes at %p (error code: %d)\n",
           SanitizerToolName, size, size, fixed_addr, GetLastError());
  return res;
}

bool MprotectNoAccess(uptr addr, uptr size) {
  DWORD old_protection;
  return VirtualProtect((LPVOID)addr, size, PAGE_NOACCESS, &old_protection);
}


void FlushUnneededShadowMemory(uptr addr, uptr size) {
  // This is almost useless on 32-bits.
  // FIXME: add madvise-analog when we move to 64-bits.
}

void NoHugePagesInRegion(uptr addr, uptr size) {
  // FIXME: probably similar to FlushUnneededShadowMemory.
}

void DontDumpShadowMemory(uptr addr, uptr length) {
  // This is almost useless on 32-bits.
  // FIXME: add madvise-analog when we move to 64-bits.
}

bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  MEMORY_BASIC_INFORMATION mbi;
  CHECK(VirtualQuery((void *)range_start, &mbi, sizeof(mbi)));
  return mbi.Protect == PAGE_NOACCESS &&
         (uptr)mbi.BaseAddress + mbi.RegionSize >= range_end;
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  UNIMPLEMENTED();
}

void *MapWritableFileToMemory(void *addr, uptr size, fd_t fd, OFF_T offset) {
  UNIMPLEMENTED();
}

static const int kMaxEnvNameLength = 128;
static const DWORD kMaxEnvValueLength = 32767;

namespace {

struct EnvVariable {
  char name[kMaxEnvNameLength];
  char value[kMaxEnvValueLength];
};

}  // namespace

static const int kEnvVariables = 5;
static EnvVariable env_vars[kEnvVariables];
static int num_env_vars;

const char *GetEnv(const char *name) {
  // Note: this implementation caches the values of the environment variables
  // and limits their quantity.
  for (int i = 0; i < num_env_vars; i++) {
    if (0 == internal_strcmp(name, env_vars[i].name))
      return env_vars[i].value;
  }
  CHECK_LT(num_env_vars, kEnvVariables);
  DWORD rv = GetEnvironmentVariableA(name, env_vars[num_env_vars].value,
                                     kMaxEnvValueLength);
  if (rv > 0 && rv < kMaxEnvValueLength) {
    CHECK_LT(internal_strlen(name), kMaxEnvNameLength);
    internal_strncpy(env_vars[num_env_vars].name, name, kMaxEnvNameLength);
    num_env_vars++;
    return env_vars[num_env_vars - 1].value;
  }
  return 0;
}

const char *GetPwd() {
  UNIMPLEMENTED();
}

u32 GetUid() {
  UNIMPLEMENTED();
}

namespace {
struct ModuleInfo {
  const char *filepath;
  uptr base_address;
  uptr end_address;
};

int CompareModulesBase(const void *pl, const void *pr) {
  const ModuleInfo *l = (ModuleInfo *)pl, *r = (ModuleInfo *)pr;
  if (l->base_address < r->base_address)
    return -1;
  return l->base_address > r->base_address;
}
}  // namespace

#ifndef SANITIZER_GO
void DumpProcessMap() {
  Report("Dumping process modules:\n");
  InternalScopedBuffer<LoadedModule> modules(kMaxNumberOfModules);
  uptr num_modules =
      GetListOfModules(modules.data(), kMaxNumberOfModules, nullptr);

  InternalScopedBuffer<ModuleInfo> module_infos(num_modules);
  for (size_t i = 0; i < num_modules; ++i) {
    module_infos[i].filepath = modules[i].full_name();
    module_infos[i].base_address = modules[i].base_address();
    module_infos[i].end_address = modules[i].ranges().next()->end;
  }
  qsort(module_infos.data(), num_modules, sizeof(ModuleInfo),
        CompareModulesBase);

  for (size_t i = 0; i < num_modules; ++i) {
    const ModuleInfo &mi = module_infos[i];
    if (mi.end_address != 0) {
      Printf("\t%p-%p %s\n", mi.base_address, mi.end_address,
             mi.filepath[0] ? mi.filepath : "[no name]");
    } else if (mi.filepath[0]) {
      Printf("\t??\?-??? %s\n", mi.filepath);
    } else {
      Printf("\t???\n");
    }
  }
}
#endif

void DisableCoreDumperIfNecessary() {
  // Do nothing.
}

void ReExec() {
  UNIMPLEMENTED();
}

void PrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
#if !SANITIZER_GO
  CovPrepareForSandboxing(args);
#endif
}

bool StackSizeIsUnlimited() {
  UNIMPLEMENTED();
}

void SetStackSizeLimitInBytes(uptr limit) {
  UNIMPLEMENTED();
}

bool AddressSpaceIsUnlimited() {
  UNIMPLEMENTED();
}

void SetAddressSpaceUnlimited() {
  UNIMPLEMENTED();
}

char *FindPathToBinary(const char *name) {
  // Nothing here for now.
  return 0;
}

bool IsPathSeparator(const char c) {
  return c == '\\' || c == '/';
}

bool IsAbsolutePath(const char *path) {
  UNIMPLEMENTED();
}

void SleepForSeconds(int seconds) {
  Sleep(seconds * 1000);
}

void SleepForMillis(int millis) {
  Sleep(millis);
}

u64 NanoTime() {
  return 0;
}

void Abort() {
  if (::IsDebuggerPresent())
    __debugbreak();
  internal__exit(3);
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  HANDLE cur_process = GetCurrentProcess();

  // Query the list of modules.  Start by assuming there are no more than 256
  // modules and retry if that's not sufficient.
  HMODULE *hmodules = 0;
  uptr modules_buffer_size = sizeof(HMODULE) * 256;
  DWORD bytes_required;
  while (!hmodules) {
    hmodules = (HMODULE *)MmapOrDie(modules_buffer_size, __FUNCTION__);
    CHECK(EnumProcessModules(cur_process, hmodules, modules_buffer_size,
                             &bytes_required));
    if (bytes_required > modules_buffer_size) {
      // Either there turned out to be more than 256 hmodules, or new hmodules
      // could have loaded since the last try.  Retry.
      UnmapOrDie(hmodules, modules_buffer_size);
      hmodules = 0;
      modules_buffer_size = bytes_required;
    }
  }

  // |num_modules| is the number of modules actually present,
  // |count| is the number of modules we return.
  size_t nun_modules = bytes_required / sizeof(HMODULE),
         count = 0;
  for (size_t i = 0; i < nun_modules && count < max_modules; ++i) {
    HMODULE handle = hmodules[i];
    MODULEINFO mi;
    if (!GetModuleInformation(cur_process, handle, &mi, sizeof(mi)))
      continue;

    char module_name[MAX_PATH];
    bool got_module_name =
        GetModuleFileNameA(handle, module_name, sizeof(module_name));
    if (!got_module_name)
      module_name[0] = '\0';

    if (filter && !filter(module_name))
      continue;

    uptr base_address = (uptr)mi.lpBaseOfDll;
    uptr end_address = (uptr)mi.lpBaseOfDll + mi.SizeOfImage;
    LoadedModule *cur_module = &modules[count];
    cur_module->set(module_name, base_address);
    // We add the whole module as one single address range.
    cur_module->addAddressRange(base_address, end_address, /*executable*/ true);
    count++;
  }
  UnmapOrDie(hmodules, modules_buffer_size);

  return count;
};

#ifndef SANITIZER_GO
// We can't use atexit() directly at __asan_init time as the CRT is not fully
// initialized at this point.  Place the functions into a vector and use
// atexit() as soon as it is ready for use (i.e. after .CRT$XIC initializers).
InternalMmapVectorNoCtor<void (*)(void)> atexit_functions;

int Atexit(void (*function)(void)) {
  atexit_functions.push_back(function);
  return 0;
}

static int RunAtexit() {
  int ret = 0;
  for (uptr i = 0; i < atexit_functions.size(); ++i) {
    ret |= atexit(atexit_functions[i]);
  }
  return ret;
}

#pragma section(".CRT$XID", long, read)  // NOLINT
static __declspec(allocate(".CRT$XID")) int (*__run_atexit)() = RunAtexit;
#endif

// ------------------ sanitizer_libc.h
fd_t OpenFile(const char *filename, FileAccessMode mode, error_t *last_error) {
  if (mode != WrOnly)
    UNIMPLEMENTED();
  fd_t res = CreateFile(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
  CHECK(res != kStdoutFd || kStdoutFd == kInvalidFd);
  CHECK(res != kStderrFd || kStderrFd == kInvalidFd);
  if (res == kInvalidFd && last_error)
    *last_error = GetLastError();
  return res;
}

void CloseFile(fd_t fd) {
  CloseHandle(fd);
}

bool ReadFromFile(fd_t fd, void *buff, uptr buff_size, uptr *bytes_read,
                  error_t *error_p) {
  UNIMPLEMENTED();
}

bool SupportsColoredOutput(fd_t fd) {
  // FIXME: support colored output.
  return false;
}

bool WriteToFile(fd_t fd, const void *buff, uptr buff_size, uptr *bytes_written,
                 error_t *error_p) {
  CHECK(fd != kInvalidFd);

  if (fd == kStdoutFd) {
    fd = GetStdHandle(STD_OUTPUT_HANDLE);
    if (fd == 0) fd = kInvalidFd;
  } else if (fd == kStderrFd) {
    fd = GetStdHandle(STD_ERROR_HANDLE);
    if (fd == 0) fd = kInvalidFd;
  }

  DWORD internal_bytes_written;
  if (fd == kInvalidFd ||
      WriteFile(fd, buff, buff_size, &internal_bytes_written, 0)) {
    if (error_p) *error_p = GetLastError();
    return false;
  } else {
    if (bytes_written) *bytes_written = internal_bytes_written;
    return true;
  }
}

bool RenameFile(const char *oldpath, const char *newpath, error_t *error_p) {
  UNIMPLEMENTED();
}

uptr internal_sched_yield() {
  Sleep(0);
  return 0;
}

void internal__exit(int exitcode) {
  ExitProcess(exitcode);
}

uptr internal_ftruncate(fd_t fd, uptr size) {
  UNIMPLEMENTED();
}

uptr GetRSS() {
  return 0;
}

void *internal_start_thread(void (*func)(void *arg), void *arg) { return 0; }
void internal_join_thread(void *th) { }

// ---------------------- BlockingMutex ---------------- {{{1
const uptr LOCK_UNINITIALIZED = 0;
const uptr LOCK_READY = (uptr)-1;

BlockingMutex::BlockingMutex(LinkerInitialized li) {
  // FIXME: see comments in BlockingMutex::Lock() for the details.
  CHECK(li == LINKER_INITIALIZED || owner_ == LOCK_UNINITIALIZED);

  CHECK(sizeof(CRITICAL_SECTION) <= sizeof(opaque_storage_));
  InitializeCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  owner_ = LOCK_READY;
}

BlockingMutex::BlockingMutex() {
  CHECK(sizeof(CRITICAL_SECTION) <= sizeof(opaque_storage_));
  InitializeCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  owner_ = LOCK_READY;
}

void BlockingMutex::Lock() {
  if (owner_ == LOCK_UNINITIALIZED) {
    // FIXME: hm, global BlockingMutex objects are not initialized?!?
    // This might be a side effect of the clang+cl+link Frankenbuild...
    new(this) BlockingMutex((LinkerInitialized)(LINKER_INITIALIZED + 1));

    // FIXME: If it turns out the linker doesn't invoke our
    // constructors, we should probably manually Lock/Unlock all the global
    // locks while we're starting in one thread to avoid double-init races.
  }
  EnterCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
  CHECK_EQ(owner_, LOCK_READY);
  owner_ = GetThreadSelf();
}

void BlockingMutex::Unlock() {
  CHECK_EQ(owner_, GetThreadSelf());
  owner_ = LOCK_READY;
  LeaveCriticalSection((LPCRITICAL_SECTION)opaque_storage_);
}

void BlockingMutex::CheckLocked() {
  CHECK_EQ(owner_, GetThreadSelf());
}

uptr GetTlsSize() {
  return 0;
}

void InitTlsSize() {
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifdef SANITIZER_GO
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#else
  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;
  *tls_addr = 0;
  *tls_size = 0;
#endif
}

#if !SANITIZER_GO
void BufferedStackTrace::SlowUnwindStack(uptr pc, u32 max_depth) {
  CHECK_GE(max_depth, 2);
  // FIXME: CaptureStackBackTrace might be too slow for us.
  // FIXME: Compare with StackWalk64.
  // FIXME: Look at LLVMUnhandledExceptionFilter in Signals.inc
  size = CaptureStackBackTrace(2, Min(max_depth, kStackTraceMax),
                               (void**)trace, 0);
  if (size == 0)
    return;

  // Skip the RTL frames by searching for the PC in the stacktrace.
  uptr pc_location = LocatePcInTrace(pc);
  PopStackFrames(pc_location);
}

void BufferedStackTrace::SlowUnwindStackWithContext(uptr pc, void *context,
                                                    u32 max_depth) {
  CONTEXT ctx = *(CONTEXT *)context;
  STACKFRAME64 stack_frame;
  memset(&stack_frame, 0, sizeof(stack_frame));
  size = 0;
#if defined(_WIN64)
  int machine_type = IMAGE_FILE_MACHINE_AMD64;
  stack_frame.AddrPC.Offset = ctx.Rip;
  stack_frame.AddrFrame.Offset = ctx.Rbp;
  stack_frame.AddrStack.Offset = ctx.Rsp;
#else
  int machine_type = IMAGE_FILE_MACHINE_I386;
  stack_frame.AddrPC.Offset = ctx.Eip;
  stack_frame.AddrFrame.Offset = ctx.Ebp;
  stack_frame.AddrStack.Offset = ctx.Esp;
#endif
  stack_frame.AddrPC.Mode = AddrModeFlat;
  stack_frame.AddrFrame.Mode = AddrModeFlat;
  stack_frame.AddrStack.Mode = AddrModeFlat;
  while (StackWalk64(machine_type, GetCurrentProcess(), GetCurrentThread(),
                     &stack_frame, &ctx, NULL, &SymFunctionTableAccess64,
                     &SymGetModuleBase64, NULL) &&
         size < Min(max_depth, kStackTraceMax)) {
    trace_buffer[size++] = (uptr)stack_frame.AddrPC.Offset;
  }
}
#endif  // #if !SANITIZER_GO

void ReportFile::Write(const char *buffer, uptr length) {
  SpinMutexLock l(mu);
  ReopenIfNecessary();
  if (!WriteToFile(fd, buffer, length)) {
    // stderr may be closed, but we may be able to print to the debugger
    // instead.  This is the case when launching a program from Visual Studio,
    // and the following routine should write to its console.
    OutputDebugStringA(buffer);
  }
}

void SetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void UnsetAlternateSignalStack() {
  // FIXME: Decide what to do on Windows.
}

void InstallDeadlySignalHandlers(SignalHandlerType handler) {
  (void)handler;
  // FIXME: Decide what to do on Windows.
}

bool IsDeadlySignal(int signum) {
  // FIXME: Decide what to do on Windows.
  return false;
}

bool IsAccessibleMemoryRange(uptr beg, uptr size) {
  SYSTEM_INFO si;
  GetNativeSystemInfo(&si);
  uptr page_size = si.dwPageSize;
  uptr page_mask = ~(page_size - 1);

  for (uptr page = beg & page_mask, end = (beg + size - 1) & page_mask;
       page <= end;) {
    MEMORY_BASIC_INFORMATION info;
    if (VirtualQuery((LPCVOID)page, &info, sizeof(info)) != sizeof(info))
      return false;

    if (info.Protect == 0 || info.Protect == PAGE_NOACCESS ||
        info.Protect == PAGE_EXECUTE)
      return false;

    if (info.RegionSize == 0)
      return false;

    page += info.RegionSize;
  }

  return true;
}

SignalContext SignalContext::Create(void *siginfo, void *context) {
  EXCEPTION_RECORD *exception_record = (EXCEPTION_RECORD*)siginfo;
  CONTEXT *context_record = (CONTEXT*)context;

  uptr pc = (uptr)exception_record->ExceptionAddress;
#ifdef _WIN64
  uptr bp = (uptr)context_record->Rbp;
  uptr sp = (uptr)context_record->Rsp;
#else
  uptr bp = (uptr)context_record->Ebp;
  uptr sp = (uptr)context_record->Esp;
#endif
  uptr access_addr = exception_record->ExceptionInformation[1];

  return SignalContext(context, access_addr, pc, sp, bp);
}

uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
  // FIXME: Actually implement this function.
  CHECK_GT(buf_len, 0);
  buf[0] = 0;
  return 0;
}

}  // namespace __sanitizer

#endif  // _WIN32
