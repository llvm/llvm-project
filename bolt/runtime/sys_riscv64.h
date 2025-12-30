#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_RISCV
#define LLVM_TOOLS_LLVM_BOLT_SYS_RISCV

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "addi sp, sp, -16\n"                                                         \
  "sd x0, 0(sp)\n"                                                             \
  "sd x1, 8(sp)\n"                                                             \
  "addi sp, sp, -16\n"                                                         \
  "sd x2, 0(sp)\n"                                                             \
  "sd x3, 8(sp)\n"                                                             \
  "addi sp, sp, -16\n"                                                         \
  "sd x4, 0(sp)\n"                                                             \
  "sd x5, 8(sp)\n"                                                             \
  "addi sp, sp, -16\n"                                                         \
  "sd x6, 0(sp)\n"                                                             \
  "sd x7, 8(sp)\n"                                                             \
  "addi sp, sp, -16\n"                                                         \
  "sd x8, 0(sp)\n"                                                             \
  "sd x9, 8(sp)\n"                                                             \
  "addi sp, sp, -16\n"                                                         \
  "sd x10, 0(sp)\n"                                                            \
  "sd x11, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x12, 0(sp)\n"                                                            \
  "sd x13, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x14, 0(sp)\n"                                                            \
  "sd x15, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x16, 0(sp)\n"                                                            \
  "sd x17, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x18, 0(sp)\n"                                                            \
  "sd x19, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x20, 0(sp)\n"                                                            \
  "sd x21, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x22, 0(sp)\n"                                                            \
  "sd x23, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x24, 0(sp)\n"                                                            \
  "sd x25, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x26, 0(sp)\n"                                                            \
  "sd x27, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x28, 0(sp)\n"                                                            \
  "sd x29, 8(sp)\n"                                                            \
  "addi sp, sp, -16\n"                                                         \
  "sd x30, 0(sp)\n"                                                            \
  "sd x31, 8(sp)\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "ld  x30, 0(sp)\n"                                                           \
  "ld  x31, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x28, 0(sp)\n"                                                           \
  "ld  x29, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x26, 0(sp)\n"                                                           \
  "ld  x27, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x24, 0(sp)\n"                                                           \
  "ld  x25, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x22, 0(sp)\n"                                                           \
  "ld  x23, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x20, 0(sp)\n"                                                           \
  "ld  x21, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x18, 0(sp)\n"                                                           \
  "ld  x19, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x16, 0(sp)\n"                                                           \
  "ld  x17, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x14, 0(sp)\n"                                                           \
  "ld  x15, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x12, 0(sp)\n"                                                           \
  "ld  x13, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x10, 0(sp)\n"                                                           \
  "ld  x11, 8(sp)\n"                                                           \
  "addi sp, sp,  16\n"                                                         \
  "ld  x8, 0(sp)\n"                                                            \
  "ld  x9, 8(sp)\n"                                                            \
  "addi sp, sp,  16\n"                                                         \
  "ld  x6, 0(sp)\n"                                                            \
  "ld  x7, 8(sp)\n"                                                            \
  "addi sp, sp,  16\n"                                                         \
  "ld  x4, 0(sp)\n"                                                            \
  "ld  x5, 8(sp)\n"                                                            \
  "addi sp, sp,  16\n"                                                         \
  "ld  x2, 0(sp)\n"                                                            \
  "ld  x3, 8(sp)\n"                                                            \
  "addi sp, sp,  16\n"                                                         \
  "ld  x0, 0(sp)\n"                                                            \
  "ld  x1, 8(sp)\n"                                                            \
  "addi sp, sp,  16\n"

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
  __asm__ volatile("lla %0, __hot_end\n\t"
                   "lui %1, %%hi(__hot_end)\n\t"
                   "addi %1, %1, %%lo(__hot_end)\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}

uint64_t __read(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t a0 __asm__("a0") = fd;
  register const void *a1 __asm__("a1") = buf;
  register uint64_t a2 __asm__("a2") = count;
  register uint64_t a7 __asm__("a7") =
      63; // Assuming 63 is the syscall number for read
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  register uint64_t a0 __asm__("a0") = fd;
  register const void *a1 __asm__("a1") = buf;
  register uint64_t a2 __asm__("a2") = count;
  register uint32_t a7 __asm__("a7") =
      64; // Assuming 64 is the syscall number for write
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
  void *ret;
  register uint64_t a0 __asm__("a0") = addr;
  register uint64_t a1 __asm__("a1") = size;
  register uint64_t a2 __asm__("a2") = prot;
  register uint64_t a3 __asm__("a3") = flags;
  register uint64_t a4 __asm__("a4") = fd;
  register uint64_t a5 __asm__("a5") = offset;
  register uint32_t a7 __asm__("a7") =
      222; // Assuming 222 is the syscall number for mmap
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5),
                         "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
  uint64_t ret;
  register void *a0 __asm__("a0") = addr;
  register uint64_t a1 __asm__("a1") = size;
  register uint32_t a7 __asm__("a7") = 215;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
  uint64_t ret;
  register uint64_t a0 __asm__("a0") = code;
  register uint32_t a7 __asm__("a7") = 94;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  register int a0 __asm__("a0") =
      -100; // Assuming -100 is an invalid file descriptor
  register const char *a1 __asm__("a1") = pathname;
  register uint64_t a2 __asm__("a2") = flags;
  register uint64_t a3 __asm__("a3") = mode;
  register uint64_t a7 __asm__("a7") =
      56; // Assuming 56 is the syscall number for open
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a7)
                       : "memory");
  return ret;
}

long __getdents64(unsigned int fd, dirent64 *dirp, size_t count) {
  long ret;
  register unsigned int a0 __asm__("a0") = fd;
  register dirent64 *a1 __asm__("a1") = dirp;
  register size_t a2 __asm__("a2") = count;
  register uint32_t a7 __asm__("a7") = 61;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  uint64_t ret;
  register int a0 __asm__("a0") = -100;
  register const char *a1 __asm__("a1") = pathname;
  register char *a2 __asm__("a2") = buf;
  register size_t a3 __asm__("a3") = bufsize;
  register uint32_t a7 __asm__("a7") = 78; // readlinkat
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a3), "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  register uint64_t a0 __asm__("a0") = fd;
  register uint64_t a1 __asm__("a1") = pos;
  register uint64_t a2 __asm__("a2") = whence;
  register uint32_t a7 __asm__("a7") = 62;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

int __ftruncate(uint64_t fd, uint64_t length) {
  int ret;
  register uint64_t a0 __asm__("a0") = fd;
  register uint64_t a1 __asm__("a1") = length;
  register uint32_t a7 __asm__("a7") = 46;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a7)
                       : "memory");
  return ret;
}

int __close(uint64_t fd) {
  int ret;
  register uint64_t a0 __asm__("a0") = fd;
  register uint32_t a7 __asm__("a7") =
      57; // Assuming 57 is the syscall number for close
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a0), "r"(a7)
                       : "memory");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  register void *a0 __asm__("a0") = addr;
  register size_t a1 __asm__("a1") = length;
  register int a2 __asm__("a2") = advice;
  register uint32_t a7 __asm__("a7") = 233;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

int __uname(struct UtsNameTy *buf) {
  int ret;
  register UtsNameTy *a0 __asm__("a0") = buf;
  register uint32_t a7 __asm__("a7") = 160;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  register const timespec *a0 __asm__("a0") = req;
  register timespec *a1 __asm__("a1") = rem;
  register uint32_t a7 __asm__("a7") = 101;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a7)
                       : "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  // clone instead of fork with flags
  // "CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD"
  register uint64_t a0 __asm__("a0") = 0x1200011;
  register uint64_t a1 __asm__("a1") = 0;
  register uint64_t a2 __asm__("a2") = 0;
  register uint64_t a3 __asm__("a3") = 0;
  register uint64_t a4 __asm__("a4") = 0;
  register uint32_t a7 __asm__("a7") = 220;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a3), "r"(a4), "r"(a7)
                       : "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  register void *a0 __asm__("a0") = addr;
  register size_t a1 __asm__("a1") = len;
  register int a2 __asm__("a2") = prot;
  register uint32_t a7 __asm__("a7") = 226;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
  register uint32_t a7 __asm__("a7") = 172;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  register uint32_t a7 __asm__("a7") = 173;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret)
                       : "r"(a7)
                       : "memory");
  return ret;
}

int __setpgid(uint64_t pid, uint64_t pgid) {
  int ret;
  register uint64_t a0 __asm__("a0") = pid;
  register uint64_t a1 __asm__("a1") = pgid;
  register uint32_t a7 __asm__("a7") = 154;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __getpgid(uint64_t pid) {
  uint64_t ret;
  register uint64_t a0 __asm__("a0") = pid;
  register uint32_t a7 __asm__("a7") = 155;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0)
                       : "r"(a7)
                       : "memory");
  return ret;
}

int __kill(uint64_t pid, int sig) {
  int ret;
  register uint64_t a0 __asm__("a0") = pid;
  register int a1 __asm__("a1") = sig;
  register uint32_t a7 __asm__("a7") = 129;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a7)
                       : "memory");
  return ret;
}

int __fsync(int fd) {
  int ret;
  register int a0 __asm__("a0") = fd;
  register uint32_t a7 __asm__("a7") = 82;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0)
                       : "r"(a7)
                       : "memory");
  return ret;
}

uint64_t __sigprocmask(int how, const void *set, void *oldset) {
  uint64_t ret;
  register int a0 __asm__("a0") = how;
  register const void *a1 __asm__("a1") = set;
  register void *a2 __asm__("a2") = oldset;
  register long a3 asm("a3") = 8;
  register uint32_t a7 __asm__("a7") = 135;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a3), "r"(a7)
                       : "memory");
  return ret;
}

int __prctl(int option, unsigned long arg2, unsigned long arg3,
            unsigned long arg4, unsigned long arg5) {
  int ret;
  register int a0 __asm__("a0") = option;
  register unsigned long a1 __asm__("a1") = arg2;
  register unsigned long a2 __asm__("a2") = arg3;
  register unsigned long a3 __asm__("a3") = arg4;
  register unsigned long a4 __asm__("a4") = arg5;
  register uint32_t a7 __asm__("a7") = 167;
  __asm__ __volatile__("ecall\n\t"
                       "mv %0, a0"
                       : "=r"(ret), "+r"(a0), "+r"(a1)
                       : "r"(a2), "r"(a3), "r"(a4), "r"(a7)
                       : "cc", "memory");
  return ret;
}
} // anonymous namespace

#endif