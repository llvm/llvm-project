//===-- sanitizer_linux_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_FREEBSD || SANITIZER_LINUX

#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_freebsd.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"

#if SANITIZER_ANDROID || SANITIZER_FREEBSD
#include <dlfcn.h>  // for dlsym()
#endif

#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>

#if SANITIZER_FREEBSD
#include <pthread_np.h>
#include <osreldate.h>
#define pthread_getattr_np pthread_attr_get_np
#endif

#if SANITIZER_LINUX
#include <sys/prctl.h>
#endif

#if SANITIZER_ANDROID
#include <android/api-level.h>
#endif

#if SANITIZER_ANDROID && __ANDROID_API__ < 21
#include <android/log.h>
#else
#include <syslog.h>
#endif

#if !SANITIZER_ANDROID
#include <elf.h>
#include <unistd.h>
#endif

namespace __sanitizer {

SANITIZER_WEAK_ATTRIBUTE int
real_sigaction(int signum, const void *act, void *oldact);

int internal_sigaction(int signum, const void *act, void *oldact) {
#if !SANITIZER_GO
  if (&real_sigaction)
    return real_sigaction(signum, act, oldact);
#endif
  return sigaction(signum, (const struct sigaction *)act,
                   (struct sigaction *)oldact);
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    MemoryMappingLayout proc_maps(/*cache_enabled*/true);
    uptr start, end, offset;
    uptr prev_end = 0;
    while (proc_maps.Next(&start, &end, &offset, nullptr, 0,
          /* protection */nullptr)) {
      if ((uptr)&rl < end)
        break;
      prev_end = end;
    }
    CHECK((uptr)&rl >= start && (uptr)&rl < end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > end - prev_end)
      stacksize = end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = end;
    *stack_bottom = end - stacksize;
    return;
  }
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  uptr stacksize = 0;
  void *stackaddr = nullptr;
  my_pthread_attr_getstack(&attr, &stackaddr, &stacksize);
  pthread_attr_destroy(&attr);

  CHECK_LE(stacksize, kMaxThreadStackSize);  // Sanity check.
  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
}

#if !SANITIZER_GO
bool SetEnv(const char *name, const char *value) {
  void *f = dlsym(RTLD_NEXT, "setenv");
  if (!f)
    return false;
  typedef int(*setenv_ft)(const char *name, const char *value, int overwrite);
  setenv_ft setenv_f;
  CHECK_EQ(sizeof(setenv_f), sizeof(f));
  internal_memcpy(&setenv_f, &f, sizeof(f));
  return setenv_f(name, value, 1) == 0;
}
#endif

bool SanitizerSetThreadName(const char *name) {
#ifdef PR_SET_NAME
  return 0 == prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);  // NOLINT
#else
  return false;
#endif
}

bool SanitizerGetThreadName(char *name, int max_len) {
#ifdef PR_GET_NAME
  char buff[17];
  if (prctl(PR_GET_NAME, (unsigned long)buff, 0, 0, 0))  // NOLINT
    return false;
  internal_strncpy(name, buff, max_len);
  name[max_len] = 0;
  return true;
#else
  return false;
#endif
}

#if !SANITIZER_FREEBSD && !SANITIZER_ANDROID && !SANITIZER_GO
static uptr g_tls_size;

#ifdef __i386__
# define DL_INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define DL_INTERNAL_FUNCTION
#endif

void InitTlsSize() {
// all current supported platforms have 16 bytes stack alignment
  const size_t kStackAlign = 16;
  typedef void (*get_tls_func)(size_t*, size_t*) DL_INTERNAL_FUNCTION;
  get_tls_func get_tls;
  void *get_tls_static_info_ptr = dlsym(RTLD_NEXT, "_dl_get_tls_static_info");
  CHECK_EQ(sizeof(get_tls), sizeof(get_tls_static_info_ptr));
  internal_memcpy(&get_tls, &get_tls_static_info_ptr,
                  sizeof(get_tls_static_info_ptr));
  CHECK_NE(get_tls, 0);
  size_t tls_size = 0;
  size_t tls_align = 0;
  get_tls(&tls_size, &tls_align);
  if (tls_align < kStackAlign)
    tls_align = kStackAlign;
  g_tls_size = RoundUpTo(tls_size, tls_align);
}
#else
void InitTlsSize() { }
#endif  // !SANITIZER_FREEBSD && !SANITIZER_ANDROID && !SANITIZER_GO

#if (defined(__x86_64__) || defined(__i386__) || defined(__mips__) \
    || defined(__aarch64__) || defined(__powerpc64__)) \
    && SANITIZER_LINUX && !SANITIZER_ANDROID
// sizeof(struct pthread) from glibc.
static atomic_uintptr_t kThreadDescriptorSize;

uptr ThreadDescriptorSize() {
  uptr val = atomic_load(&kThreadDescriptorSize, memory_order_relaxed);
  if (val)
    return val;
#if defined(__x86_64__) || defined(__i386__)
#ifdef _CS_GNU_LIBC_VERSION
  char buf[64];
  uptr len = confstr(_CS_GNU_LIBC_VERSION, buf, sizeof(buf));
  if (len < sizeof(buf) && internal_strncmp(buf, "glibc 2.", 8) == 0) {
    char *end;
    int minor = internal_simple_strtoll(buf + 8, &end, 10);
    if (end != buf + 8 && (*end == '\0' || *end == '.')) {
      int patch = 0;
      if (*end == '.')
        // strtoll will return 0 if no valid conversion could be performed
        patch = internal_simple_strtoll(end + 1, nullptr, 10);

      /* sizeof(struct pthread) values from various glibc versions.  */
      if (SANITIZER_X32)
        val = 1728;  // Assume only one particular version for x32.
      else if (minor <= 3)
        val = FIRST_32_SECOND_64(1104, 1696);
      else if (minor == 4)
        val = FIRST_32_SECOND_64(1120, 1728);
      else if (minor == 5)
        val = FIRST_32_SECOND_64(1136, 1728);
      else if (minor <= 9)
        val = FIRST_32_SECOND_64(1136, 1712);
      else if (minor == 10)
        val = FIRST_32_SECOND_64(1168, 1776);
      else if (minor == 11 || (minor == 12 && patch == 1))
        val = FIRST_32_SECOND_64(1168, 2288);
      else if (minor <= 13)
        val = FIRST_32_SECOND_64(1168, 2304);
      else
        val = FIRST_32_SECOND_64(1216, 2304);
    }
    if (val)
      atomic_store(&kThreadDescriptorSize, val, memory_order_relaxed);
    return val;
  }
#endif
#elif defined(__mips__)
  // TODO(sagarthakur): add more values as per different glibc versions.
  val = FIRST_32_SECOND_64(1152, 1776);
  if (val)
    atomic_store(&kThreadDescriptorSize, val, memory_order_relaxed);
  return val;
#elif defined(__aarch64__)
  // The sizeof (struct pthread) is the same from GLIBC 2.17 to 2.22.
  val = 1776;
  atomic_store(&kThreadDescriptorSize, val, memory_order_relaxed);
  return val;
#elif defined(__powerpc64__)
  val = 1776; // from glibc.ppc64le 2.20-8.fc21
  atomic_store(&kThreadDescriptorSize, val, memory_order_relaxed);
  return val;
#endif
  return 0;
}

// The offset at which pointer to self is located in the thread descriptor.
const uptr kThreadSelfOffset = FIRST_32_SECOND_64(8, 16);

uptr ThreadSelfOffset() {
  return kThreadSelfOffset;
}

#if defined(__mips__) || defined(__powerpc64__)
// TlsPreTcbSize includes size of struct pthread_descr and size of tcb
// head structure. It lies before the static tls blocks.
static uptr TlsPreTcbSize() {
# if defined(__mips__)
  const uptr kTcbHead = 16; // sizeof (tcbhead_t)
# elif defined(__powerpc64__)
  const uptr kTcbHead = 88; // sizeof (tcbhead_t)
# endif
  const uptr kTlsAlign = 16;
  const uptr kTlsPreTcbSize =
    (ThreadDescriptorSize() + kTcbHead + kTlsAlign - 1) & ~(kTlsAlign - 1);
  InitTlsSize();
  g_tls_size = (g_tls_size + kTlsPreTcbSize + kTlsAlign -1) & ~(kTlsAlign - 1);
  return kTlsPreTcbSize;
}
#endif

uptr ThreadSelf() {
  uptr descr_addr;
# if defined(__i386__)
  asm("mov %%gs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
# elif defined(__x86_64__)
  asm("mov %%fs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
# elif defined(__mips__)
  // MIPS uses TLS variant I. The thread pointer (in hardware register $29)
  // points to the end of the TCB + 0x7000. The pthread_descr structure is
  // immediately in front of the TCB. TlsPreTcbSize() includes the size of the
  // TCB and the size of pthread_descr.
  const uptr kTlsTcbOffset = 0x7000;
  uptr thread_pointer;
  asm volatile(".set push;\
                .set mips64r2;\
                rdhwr %0,$29;\
                .set pop" : "=r" (thread_pointer));
  descr_addr = thread_pointer - kTlsTcbOffset - TlsPreTcbSize();
# elif defined(__aarch64__)
  descr_addr = reinterpret_cast<uptr>(__builtin_thread_pointer());
# elif defined(__powerpc64__)
  // PPC64LE uses TLS variant I. The thread pointer (in GPR 13)
  // points to the end of the TCB + 0x7000. The pthread_descr structure is
  // immediately in front of the TCB. TlsPreTcbSize() includes the size of the
  // TCB and the size of pthread_descr.
  const uptr kTlsTcbOffset = 0x7000;
  uptr thread_pointer;
  asm("addi %0,13,%1" : "=r"(thread_pointer) : "I"(-kTlsTcbOffset));
  descr_addr = thread_pointer - TlsPreTcbSize();
# else
#  error "unsupported CPU arch"
# endif
  return descr_addr;
}
#endif  // (x86_64 || i386 || MIPS) && SANITIZER_LINUX

#if SANITIZER_FREEBSD
static void **ThreadSelfSegbase() {
  void **segbase = 0;
# if defined(__i386__)
  // sysarch(I386_GET_GSBASE, segbase);
  __asm __volatile("mov %%gs:0, %0" : "=r" (segbase));
# elif defined(__x86_64__)
  // sysarch(AMD64_GET_FSBASE, segbase);
  __asm __volatile("movq %%fs:0, %0" : "=r" (segbase));
# else
#  error "unsupported CPU arch for FreeBSD platform"
# endif
  return segbase;
}

uptr ThreadSelf() {
  return (uptr)ThreadSelfSegbase()[2];
}
#endif  // SANITIZER_FREEBSD

#if !SANITIZER_GO
static void GetTls(uptr *addr, uptr *size) {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
# if defined(__x86_64__) || defined(__i386__)
  *addr = ThreadSelf();
  *size = GetTlsSize();
  *addr -= *size;
  *addr += ThreadDescriptorSize();
# elif defined(__mips__) || defined(__aarch64__) || defined(__powerpc64__)
  *addr = ThreadSelf();
  *size = GetTlsSize();
# else
  *addr = 0;
  *size = 0;
# endif
#elif SANITIZER_FREEBSD
  void** segbase = ThreadSelfSegbase();
  *addr = 0;
  *size = 0;
  if (segbase != 0) {
    // tcbalign = 16
    // tls_size = round(tls_static_space, tcbalign);
    // dtv = segbase[1];
    // dtv[2] = segbase - tls_static_space;
    void **dtv = (void**) segbase[1];
    *addr = (uptr) dtv[2];
    *size = (*addr == 0) ? 0 : ((uptr) segbase[0] - (uptr) dtv[2]);
  }
#elif SANITIZER_ANDROID
  *addr = 0;
  *size = 0;
#else
# error "Unknown OS"
#endif
}
#endif

#if !SANITIZER_GO
uptr GetTlsSize() {
#if SANITIZER_FREEBSD || SANITIZER_ANDROID
  uptr addr, size;
  GetTls(&addr, &size);
  return size;
#else
  return g_tls_size;
#endif
}
#endif

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#if SANITIZER_GO
  // Stub implementation for Go.
  *stk_addr = *stk_size = *tls_addr = *tls_size = 0;
#else
  GetTls(tls_addr, tls_size);

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;

  if (!main) {
    // If stack and tls intersect, make them non-intersecting.
    if (*tls_addr > *stk_addr && *tls_addr < *stk_addr + *stk_size) {
      CHECK_GT(*tls_addr + *tls_size, *stk_addr);
      CHECK_LE(*tls_addr + *tls_size, *stk_addr + *stk_size);
      *stk_size -= *tls_size;
      *tls_addr = *stk_addr + *stk_size;
    }
  }
#endif
}

# if !SANITIZER_FREEBSD
typedef ElfW(Phdr) Elf_Phdr;
# elif SANITIZER_WORDSIZE == 32 && __FreeBSD_version <= 902001  // v9.2
#  define Elf_Phdr XElf32_Phdr
#  define dl_phdr_info xdl_phdr_info
#  define dl_iterate_phdr(c, b) xdl_iterate_phdr((c), (b))
# endif

struct DlIteratePhdrData {
  InternalMmapVector<LoadedModule> *modules;
  bool first;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  InternalScopedString module_name(kMaxPathLength);
  if (data->first) {
    data->first = false;
    // First module is the binary itself.
    ReadBinaryNameCached(module_name.data(), module_name.size());
  } else if (info->dlpi_name) {
    module_name.append("%s", info->dlpi_name);
  }
  if (module_name[0] == '\0')
    return 0;
  LoadedModule cur_module;
  cur_module.set(module_name.data(), info->dlpi_addr);
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      bool executable = phdr->p_flags & PF_X;
      cur_module.addAddressRange(cur_beg, cur_end, executable);
    }
  }
  data->modules->push_back(cur_module);
  return 0;
}

#if SANITIZER_ANDROID && __ANDROID_API__ < 21
extern "C" __attribute__((weak)) int dl_iterate_phdr(
    int (*)(struct dl_phdr_info *, size_t, void *), void *);
#endif

void ListOfModules::init() {
  clear();
#if SANITIZER_ANDROID && __ANDROID_API__ <= 22
  u32 api_level = AndroidGetApiLevel();
  // Fall back to /proc/maps if dl_iterate_phdr is unavailable or broken.
  // The runtime check allows the same library to work with
  // both K and L (and future) Android releases.
  if (api_level <= ANDROID_LOLLIPOP_MR1) { // L or earlier
    MemoryMappingLayout memory_mapping(false);
    memory_mapping.DumpListOfModules(&modules_);
    return;
  }
#endif
  DlIteratePhdrData data = {&modules_, true};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
}

// getrusage does not give us the current RSS, only the max RSS.
// Still, this is better than nothing if /proc/self/statm is not available
// for some reason, e.g. due to a sandbox.
static uptr GetRSSFromGetrusage() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage))  // Failed, probably due to a sandbox.
    return 0;
  return usage.ru_maxrss << 10;  // ru_maxrss is in Kb.
}

uptr GetRSS() {
  if (!common_flags()->can_use_proc_maps_statm)
    return GetRSSFromGetrusage();
  fd_t fd = OpenFile("/proc/self/statm", RdOnly);
  if (fd == kInvalidFd)
    return GetRSSFromGetrusage();
  char buf[64];
  uptr len = internal_read(fd, buf, sizeof(buf) - 1);
  internal_close(fd);
  if ((sptr)len <= 0)
    return 0;
  buf[len] = 0;
  // The format of the file is:
  // 1084 89 69 11 0 79 0
  // We need the second number which is RSS in pages.
  char *pos = buf;
  // Skip the first number.
  while (*pos >= '0' && *pos <= '9')
    pos++;
  // Skip whitespaces.
  while (!(*pos >= '0' && *pos <= '9') && *pos != 0)
    pos++;
  // Read the number.
  uptr rss = 0;
  while (*pos >= '0' && *pos <= '9')
    rss = rss * 10 + *pos++ - '0';
  return rss * GetPageSizeCached();
}

// 64-bit Android targets don't provide the deprecated __android_log_write.
// Starting with the L release, syslog() works and is preferable to
// __android_log_write.
#if SANITIZER_LINUX

#if SANITIZER_ANDROID
static atomic_uint8_t android_log_initialized;

void AndroidLogInit() {
  atomic_store(&android_log_initialized, 1, memory_order_release);
}

static bool ShouldLogAfterPrintf() {
  return atomic_load(&android_log_initialized, memory_order_acquire);
}
#else
void AndroidLogInit() {}

static bool ShouldLogAfterPrintf() { return true; }
#endif  // SANITIZER_ANDROID

void WriteOneLineToSyslog(const char *s) {
#if SANITIZER_ANDROID &&__ANDROID_API__ < 21
  __android_log_write(ANDROID_LOG_INFO, NULL, s);
#else
  syslog(LOG_INFO, "%s", s);
#endif
}

void LogMessageOnPrintf(const char *str) {
  if (common_flags()->log_to_syslog && ShouldLogAfterPrintf())
    WriteToSyslog(str);
}

#endif // SANITIZER_LINUX

} // namespace __sanitizer

#endif // SANITIZER_FREEBSD || SANITIZER_LINUX
