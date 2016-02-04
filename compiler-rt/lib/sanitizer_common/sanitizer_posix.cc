//===-- sanitizer_posix.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements POSIX-specific functions from
// sanitizer_posix.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_POSIX

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_posix.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>

#if SANITIZER_LINUX
#include <sys/utsname.h>
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
#include <sys/personality.h>
#endif

#if SANITIZER_FREEBSD
// The MAP_NORESERVE define has been removed in FreeBSD 11.x, and even before
// that, it was never implemented.  So just define it to zero.
#undef  MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

namespace __sanitizer {

// ------------- sanitizer_common.h
uptr GetMmapGranularity() {
  return GetPageSize();
}

#if SANITIZER_WORDSIZE == 32
// Take care of unusable kernel area in top gigabyte.
static uptr GetKernelAreaSize() {
#if SANITIZER_LINUX && !SANITIZER_X32
  const uptr gbyte = 1UL << 30;

  // Firstly check if there are writable segments
  // mapped to top gigabyte (e.g. stack).
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr end, prot;
  while (proc_maps.Next(/*start*/nullptr, &end,
                        /*offset*/nullptr, /*filename*/nullptr,
                        /*filename_size*/0, &prot)) {
    if ((end >= 3 * gbyte)
        && (prot & MemoryMappingLayout::kProtectionWrite) != 0)
      return 0;
  }

#if !SANITIZER_ANDROID
  // Even if nothing is mapped, top Gb may still be accessible
  // if we are running on 64-bit kernel.
  // Uname may report misleading results if personality type
  // is modified (e.g. under schroot) so check this as well.
  struct utsname uname_info;
  int pers = personality(0xffffffffUL);
  if (!(pers & PER_MASK)
      && uname(&uname_info) == 0
      && internal_strstr(uname_info.machine, "64"))
    return 0;
#endif  // SANITIZER_ANDROID

  // Top gigabyte is reserved for kernel.
  return gbyte;
#else
  return 0;
#endif  // SANITIZER_LINUX && !SANITIZER_X32
}
#endif  // SANITIZER_WORDSIZE == 32

uptr GetMaxVirtualAddress() {
#if SANITIZER_WORDSIZE == 64
# if defined(__aarch64__) && SANITIZER_IOS && !SANITIZER_IOSSIM
  // Ideally, we would derive the upper bound from MACH_VM_MAX_ADDRESS. The
  // upper bound can change depending on the device.
  return 0x200000000 - 1;
# elif defined(__powerpc64__) || defined(__aarch64__)
  // On PowerPC64 we have two different address space layouts: 44- and 46-bit.
  // We somehow need to figure out which one we are using now and choose
  // one of 0x00000fffffffffffUL and 0x00003fffffffffffUL.
  // Note that with 'ulimit -s unlimited' the stack is moved away from the top
  // of the address space, so simply checking the stack address is not enough.
  // This should (does) work for both PowerPC64 Endian modes.
  // Similarly, aarch64 has multiple address space layouts: 39, 42 and 47-bit.
  return (1ULL << (MostSignificantSetBitIndex(GET_CURRENT_FRAME()) + 1)) - 1;
# elif defined(__mips64)
  return (1ULL << 40) - 1;  // 0x000000ffffffffffUL;
# else
  return (1ULL << 47) - 1;  // 0x00007fffffffffffUL;
# endif
#else  // SANITIZER_WORDSIZE == 32
  uptr res = (1ULL << 32) - 1;  // 0xffffffff;
  if (!common_flags()->full_address_space)
    res -= GetKernelAreaSize();
  CHECK_LT(reinterpret_cast<uptr>(&res), res);
  return res;
#endif  // SANITIZER_WORDSIZE
}

void *MmapOrDie(uptr size, const char *mem_type, bool raw_report) {
  size = RoundUpTo(size, GetPageSizeCached());
  uptr res = internal_mmap(nullptr, size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANON, -1, 0);
  int reserrno;
  if (internal_iserror(res, &reserrno))
    ReportMmapFailureAndDie(size, mem_type, "allocate", reserrno, raw_report);
  IncreaseTotalMmap(size);
  return (void *)res;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  uptr res = internal_munmap(addr, size);
  if (internal_iserror(res)) {
    Report("ERROR: %s failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           SanitizerToolName, size, size, addr);
    CHECK("unable to unmap" && 0);
  }
  DecreaseTotalMmap(size);
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap(nullptr,
                         RoundUpTo(size, PageSize),
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
                         -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno))
    ReportMmapFailureAndDie(size, mem_type, "allocate noreserve", reserrno);
  IncreaseTotalMmap(size);
  return (void *)p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno)) {
    char mem_type[30];
    internal_snprintf(mem_type, sizeof(mem_type), "memory at address 0x%zx",
                      fixed_addr);
    ReportMmapFailureAndDie(size, mem_type, "allocate", reserrno);
  }
  IncreaseTotalMmap(size);
  return (void *)p;
}

bool MprotectNoAccess(uptr addr, uptr size) {
  return 0 == internal_mprotect((void*)addr, size, PROT_NONE);
}

bool MprotectReadOnly(uptr addr, uptr size) {
  return 0 == internal_mprotect((void *)addr, size, PROT_READ);
}

fd_t OpenFile(const char *filename, FileAccessMode mode, error_t *errno_p) {
  int flags;
  switch (mode) {
    case RdOnly: flags = O_RDONLY; break;
    case WrOnly: flags = O_WRONLY | O_CREAT; break;
    case RdWr: flags = O_RDWR | O_CREAT; break;
  }
  fd_t res = internal_open(filename, flags, 0660);
  if (internal_iserror(res, errno_p))
    return kInvalidFd;
  return res;
}

void CloseFile(fd_t fd) {
  internal_close(fd);
}

bool ReadFromFile(fd_t fd, void *buff, uptr buff_size, uptr *bytes_read,
                  error_t *error_p) {
  uptr res = internal_read(fd, buff, buff_size);
  if (internal_iserror(res, error_p))
    return false;
  if (bytes_read)
    *bytes_read = res;
  return true;
}

bool WriteToFile(fd_t fd, const void *buff, uptr buff_size, uptr *bytes_written,
                 error_t *error_p) {
  uptr res = internal_write(fd, buff, buff_size);
  if (internal_iserror(res, error_p))
    return false;
  if (bytes_written)
    *bytes_written = res;
  return true;
}

bool RenameFile(const char *oldpath, const char *newpath, error_t *error_p) {
  uptr res = internal_rename(oldpath, newpath);
  return !internal_iserror(res, error_p);
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  fd_t fd = OpenFile(file_name, RdOnly);
  CHECK(fd != kInvalidFd);
  uptr fsize = internal_filesize(fd);
  CHECK_NE(fsize, (uptr)-1);
  CHECK_GT(fsize, 0);
  *buff_size = RoundUpTo(fsize, GetPageSizeCached());
  uptr map = internal_mmap(nullptr, *buff_size, PROT_READ, MAP_PRIVATE, fd, 0);
  return internal_iserror(map) ? nullptr : (void *)map;
}

void *MapWritableFileToMemory(void *addr, uptr size, fd_t fd, OFF_T offset) {
  uptr flags = MAP_SHARED;
  if (addr) flags |= MAP_FIXED;
  uptr p = internal_mmap(addr, size, PROT_READ | PROT_WRITE, flags, fd, offset);
  int mmap_errno = 0;
  if (internal_iserror(p, &mmap_errno)) {
    Printf("could not map writable file (%d, %lld, %zu): %zd, errno: %d\n",
           fd, (long long)offset, size, p, mmap_errno);
    return nullptr;
  }
  return (void *)p;
}

static inline bool IntervalsAreSeparate(uptr start1, uptr end1,
                                        uptr start2, uptr end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  while (proc_maps.Next(&start, &end,
                        /*offset*/nullptr, /*filename*/nullptr,
                        /*filename_size*/0, /*protection*/nullptr)) {
    if (start == end) continue;  // Empty range.
    CHECK_NE(0, end);
    if (!IntervalsAreSeparate(start, end - 1, range_start, range_end))
      return false;
  }
  return true;
}

void DumpProcessMap() {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  const sptr kBufSize = 4095;
  char *filename = (char*)MmapOrDie(kBufSize, __func__);
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&start, &end, /* file_offset */nullptr,
                        filename, kBufSize, /* protection */nullptr)) {
    Printf("\t%p-%p\t%s\n", (void*)start, (void*)end, filename);
  }
  Report("End of process memory map.\n");
  UnmapOrDie(filename, kBufSize);
}

const char *GetPwd() {
  return GetEnv("PWD");
}

bool IsPathSeparator(const char c) {
  return c == '/';
}

bool IsAbsolutePath(const char *path) {
  return path != nullptr && IsPathSeparator(path[0]);
}

void ReportFile::Write(const char *buffer, uptr length) {
  SpinMutexLock l(mu);
  static const char *kWriteError =
      "ReportFile::Write() can't output requested buffer!\n";
  ReopenIfNecessary();
  if (length != internal_write(fd, buffer, length)) {
    internal_write(fd, kWriteError, internal_strlen(kWriteError));
    Die();
  }
}

bool GetCodeRangeForFile(const char *module, uptr *start, uptr *end) {
  uptr s, e, off, prot;
  InternalScopedString buff(kMaxPathLength);
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  while (proc_maps.Next(&s, &e, &off, buff.data(), buff.size(), &prot)) {
    if ((prot & MemoryMappingLayout::kProtectionExecute) != 0
        && internal_strcmp(module, buff.data()) == 0) {
      *start = s;
      *end = e;
      return true;
    }
  }
  return false;
}

SignalContext SignalContext::Create(void *siginfo, void *context) {
  auto si = (siginfo_t*)siginfo;
  uptr addr = (uptr)si->si_addr;
  uptr pc, sp, bp;
  GetPcSpBp(context, &pc, &sp, &bp);
  bool is_write = GetSigContextWriteFlag(context);
  bool is_memory_access = si->si_signo == SIGSEGV;
  return SignalContext(context, addr, pc, sp, bp, is_memory_access, is_write);
}

} // namespace __sanitizer

#endif // SANITIZER_POSIX
