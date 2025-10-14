#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_AARCH64
#define LLVM_TOOLS_LLVM_BOLT_SYS_AARCH64

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "stp x0, x1, [sp, #-16]!\n"                                                  \
  "stp x2, x3, [sp, #-16]!\n"                                                  \
  "stp x4, x5, [sp, #-16]!\n"                                                  \
  "stp x6, x7, [sp, #-16]!\n"                                                  \
  "stp x8, x9, [sp, #-16]!\n"                                                  \
  "stp x10, x11, [sp, #-16]!\n"                                                \
  "stp x12, x13, [sp, #-16]!\n"                                                \
  "stp x14, x15, [sp, #-16]!\n"                                                \
  "stp x16, x17, [sp, #-16]!\n"                                                \
  "stp x18, x19, [sp, #-16]!\n"                                                \
  "stp x20, x21, [sp, #-16]!\n"                                                \
  "stp x22, x23, [sp, #-16]!\n"                                                \
  "stp x24, x25, [sp, #-16]!\n"                                                \
  "stp x26, x27, [sp, #-16]!\n"                                                \
  "stp x28, x29, [sp, #-16]!\n"                                                \
  "str x30, [sp,#-16]!\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "ldr x30, [sp], #16\n"                                                       \
  "ldp x28, x29, [sp], #16\n"                                                  \
  "ldp x26, x27, [sp], #16\n"                                                  \
  "ldp x24, x25, [sp], #16\n"                                                  \
  "ldp x22, x23, [sp], #16\n"                                                  \
  "ldp x20, x21, [sp], #16\n"                                                  \
  "ldp x18, x19, [sp], #16\n"                                                  \
  "ldp x16, x17, [sp], #16\n"                                                  \
  "ldp x14, x15, [sp], #16\n"                                                  \
  "ldp x12, x13, [sp], #16\n"                                                  \
  "ldp x10, x11, [sp], #16\n"                                                  \
  "ldp x8, x9, [sp], #16\n"                                                    \
  "ldp x6, x7, [sp], #16\n"                                                    \
  "ldp x4, x5, [sp], #16\n"                                                    \
  "ldp x2, x3, [sp], #16\n"                                                    \
  "ldp x0, x1, [sp], #16\n"

// Anonymous namespace covering everything but our library entry point
namespace {

// Get the difference between runtime address of .text section and
// static address in section header table. Can be extracted from arbitrary
// pc value recorded at runtime to get the corresponding static address, which
// in turn can be used to search for indirect call description. Needed because
// indirect call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("b .instr%=\n\t"
                   ".StaticAddr%=:\n\t"
                   ".dword __hot_end\n\t"
                   ".instr%=:\n\t"
                   "ldr %0, .StaticAddr%=\n\t"
                   "adrp %1, __hot_end\n\t"
                   "add %1, %1, :lo12:__hot_end\n\t"
                   : "=r"(StaticAddr), "=r"(DynAddr));
  return DynAddr - StaticAddr;
}

uint64_t __read(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t x0 __asm__("x0") = fd;
  register const void *x1 __asm__("x1") = buf;
  register uint64_t x2 __asm__("x2") = count;
  register uint32_t w8 __asm__("w8") = 63;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t x0 __asm__("x0") = fd;
  register const void *x1 __asm__("x1") = buf;
  register uint64_t x2 __asm__("x2") = count;
  register uint32_t w8 __asm__("w8") = 64;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
  void *ret;
  register uint64_t x0 __asm__("x0") = addr;
  register uint64_t x1 __asm__("x1") = size;
  register uint64_t x2 __asm__("x2") = prot;
  register uint64_t x3 __asm__("x3") = flags;
  register uint64_t x4 __asm__("x4") = fd;
  register uint64_t x5 __asm__("x5") = offset;
  register uint32_t w8 __asm__("w8") = 222;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(x4), "r"(x5), "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
  uint64_t ret;
  register void *x0 __asm__("x0") = addr;
  register uint64_t x1 __asm__("x1") = size;
  register uint32_t w8 __asm__("w8") = 215;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
  uint64_t ret;
  register uint64_t x0 __asm__("x0") = code;
  register uint32_t w8 __asm__("w8") = 94;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0)
                       : "r"(w8)
                       : "cc", "memory", "x1");
  return ret;
}

uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  register int x0 __asm__("x0") = -100;
  register const char *x1 __asm__("x1") = pathname;
  register uint64_t x2 __asm__("x2") = flags;
  register uint64_t x3 __asm__("x3") = mode;
  register uint32_t w8 __asm__("w8") = 56;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(w8)
                       : "cc", "memory");
  return ret;
}

long __getdents64(unsigned int fd, dirent64 *dirp, size_t count) {
  long ret;
  register unsigned int x0 __asm__("x0") = fd;
  register dirent64 *x1 __asm__("x1") = dirp;
  register size_t x2 __asm__("x2") = count;
  register uint32_t w8 __asm__("w8") = 61;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  uint64_t ret;
  register int x0 __asm__("x0") = -100;
  register const char *x1 __asm__("x1") = pathname;
  register char *x2 __asm__("x2") = buf;
  register size_t x3 __asm__("x3") = bufsize;
  register uint32_t w8 __asm__("w8") = 78; // readlinkat
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  register uint64_t x0 __asm__("x0") = fd;
  register uint64_t x1 __asm__("x1") = pos;
  register uint64_t x2 __asm__("x2") = whence;
  register uint32_t w8 __asm__("w8") = 62;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __ftruncate(uint64_t fd, uint64_t length) {
  int ret;
  register uint64_t x0 __asm__("x0") = fd;
  register uint64_t x1 __asm__("x1") = length;
  register uint32_t w8 __asm__("w8") = 46;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __close(uint64_t fd) {
  int ret;
  register uint64_t x0 __asm__("x0") = fd;
  register uint32_t w8 __asm__("w8") = 57;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0)
                       : "r"(w8)
                       : "cc", "memory", "x1");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  register void *x0 __asm__("x0") = addr;
  register size_t x1 __asm__("x1") = length;
  register int x2 __asm__("x2") = advice;
  register uint32_t w8 __asm__("w8") = 233;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __uname(struct UtsNameTy *buf) {
  int ret;
  register UtsNameTy *x0 __asm__("x0") = buf;
  register uint32_t w8 __asm__("w8") = 160;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0)
                       : "r"(w8)
                       : "cc", "memory", "x1");
  return ret;
}

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  register const timespec *x0 __asm__("x0") = req;
  register timespec *x1 __asm__("x1") = rem;
  register uint32_t w8 __asm__("w8") = 101;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(w8)
                       : "cc", "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  // clone instead of fork with flags
  // "CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD"
  register uint64_t x0 __asm__("x0") = 0x1200011;
  register uint64_t x1 __asm__("x1") = 0;
  register uint64_t x2 __asm__("x2") = 0;
  register uint64_t x3 __asm__("x3") = 0;
  register uint64_t x4 __asm__("x4") = 0;
  register uint32_t w8 __asm__("w8") = 220;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(x4), "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  register void *x0 __asm__("x0") = addr;
  register size_t x1 __asm__("x1") = len;
  register int x2 __asm__("x2") = prot;
  register uint32_t w8 __asm__("w8") = 226;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
  register uint32_t w8 __asm__("w8") = 172;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret)
                       : "r"(w8)
                       : "cc", "memory", "x0", "x1");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  register uint32_t w8 __asm__("w8") = 173;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret)
                       : "r"(w8)
                       : "cc", "memory", "x0", "x1");
  return ret;
}

int __setpgid(uint64_t pid, uint64_t pgid) {
  int ret;
  register uint64_t x0 __asm__("x0") = pid;
  register uint64_t x1 __asm__("x1") = pgid;
  register uint32_t w8 __asm__("w8") = 154;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(w8)
                       : "cc", "memory");
  return ret;
}

uint64_t __getpgid(uint64_t pid) {
  uint64_t ret;
  register uint64_t x0 __asm__("x0") = pid;
  register uint32_t w8 __asm__("w8") = 155;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0)
                       : "r"(w8)
                       : "cc", "memory", "x1");
  return ret;
}

int __kill(uint64_t pid, int sig) {
  int ret;
  register uint64_t x0 __asm__("x0") = pid;
  register int x1 __asm__("x1") = sig;
  register uint32_t w8 __asm__("w8") = 129;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __fsync(int fd) {
  int ret;
  register int x0 __asm__("x0") = fd;
  register uint32_t w8 __asm__("w8") = 82;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0)
                       : "r"(w8)
                       : "cc", "memory", "x1");
  return ret;
}

uint64_t __sigprocmask(int how, const void *set, void *oldset) {
  uint64_t ret;
  register int x0 __asm__("x0") = how;
  register const void *x1 __asm__("x1") = set;
  register void *x2 __asm__("x2") = oldset;
  register long x3 asm("x3") = 8;
  register uint32_t w8 __asm__("w8") = 135;
  __asm__ __volatile__("svc #0\n"
                       "mov %0, x0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(w8)
                       : "cc", "memory");
  return ret;
}

int __prctl(int option, unsigned long arg2, unsigned long arg3,
            unsigned long arg4, unsigned long arg5) {
  int ret;
  register int x0 __asm__("x0") = option;
  register unsigned long x1 __asm__("x1") = arg2;
  register unsigned long x2 __asm__("x2") = arg3;
  register unsigned long x3 __asm__("x3") = arg4;
  register unsigned long x4 __asm__("x4") = arg5;
  register uint32_t w8 __asm__("w8") = 167;
  __asm__ __volatile__("svc #0\n"
                       "mov %w0, w0"
                       : "=r"(ret), "+r"(x0), "+r"(x1)
                       : "r"(x2), "r"(x3), "r"(x4), "r"(w8)
                       : "cc", "memory");
  return ret;
}

} // anonymous namespace

#endif
