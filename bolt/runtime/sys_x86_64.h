#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_X86_64
#define LLVM_TOOLS_LLVM_BOLT_SYS_X86_64

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "push %%rax\n"                                                               \
  "push %%rbx\n"                                                               \
  "push %%rcx\n"                                                               \
  "push %%rdx\n"                                                               \
  "push %%rdi\n"                                                               \
  "push %%rsi\n"                                                               \
  "push %%rbp\n"                                                               \
  "push %%r8\n"                                                                \
  "push %%r9\n"                                                                \
  "push %%r10\n"                                                               \
  "push %%r11\n"                                                               \
  "push %%r12\n"                                                               \
  "push %%r13\n"                                                               \
  "push %%r14\n"                                                               \
  "push %%r15\n"                                                               \
  "sub $8, %%rsp\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "add $8, %%rsp\n"                                                            \
  "pop %%r15\n"                                                                \
  "pop %%r14\n"                                                                \
  "pop %%r13\n"                                                                \
  "pop %%r12\n"                                                                \
  "pop %%r11\n"                                                                \
  "pop %%r10\n"                                                                \
  "pop %%r9\n"                                                                 \
  "pop %%r8\n"                                                                 \
  "pop %%rbp\n"                                                                \
  "pop %%rsi\n"                                                                \
  "pop %%rdi\n"                                                                \
  "pop %%rdx\n"                                                                \
  "pop %%rcx\n"                                                                \
  "pop %%rbx\n"                                                                \
  "pop %%rax\n"

namespace {

// Get the difference between runtime addrress of .text section and
// static address in section header table. Can be extracted from arbitrary
// pc value recorded at runtime to get the corresponding static address, which
// in turn can be used to search for indirect call description. Needed because
// indirect call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("leaq __hot_end(%%rip), %0\n\t"
                   "movabsq $__hot_end, %1\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

uint64_t __read(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
#if defined(__APPLE__)
#define READ_SYSCALL 0x2000003
#else
#define READ_SYSCALL 0
#endif
  __asm__ __volatile__("movq $" STRINGIFY(READ_SYSCALL) ", %%rax\n"
                                                        "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(buf), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
#if defined(__APPLE__)
#define WRITE_SYSCALL 0x2000004
#else
#define WRITE_SYSCALL 1
#endif
  __asm__ __volatile__("movq $" STRINGIFY(WRITE_SYSCALL) ", %%rax\n"
                                                         "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(buf), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
#if defined(__APPLE__)
#define MMAP_SYSCALL 0x20000c5
#else
#define MMAP_SYSCALL 9
#endif
  void *ret;
  register uint64_t r8 asm("r8") = fd;
  register uint64_t r9 asm("r9") = offset;
  register uint64_t r10 asm("r10") = flags;
  __asm__ __volatile__("movq $" STRINGIFY(MMAP_SYSCALL) ", %%rax\n"
                                                        "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size), "d"(prot), "r"(r10), "r"(r8),
                         "r"(r9)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
#if defined(__APPLE__)
#define MUNMAP_SYSCALL 0x2000049
#else
#define MUNMAP_SYSCALL 11
#endif
  uint64_t ret;
  __asm__ __volatile__("movq $" STRINGIFY(MUNMAP_SYSCALL) ", %%rax\n"
                                                          "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __sigprocmask(int how, const void *set, void *oldset) {
#if defined(__APPLE__)
#define SIGPROCMASK_SYSCALL 0x2000030
#else
#define SIGPROCMASK_SYSCALL 14
#endif
  uint64_t ret;
  register long r10 asm("r10") = sizeof(uint64_t);
  __asm__ __volatile__("movq $" STRINGIFY(SIGPROCMASK_SYSCALL) ", %%rax\n"
                                                               "syscall\n"
                       : "=a"(ret)
                       : "D"(how), "S"(set), "d"(oldset), "r"(r10)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
#if defined(__APPLE__)
#define GETPID_SYSCALL 20
#else
#define GETPID_SYSCALL 39
#endif
  __asm__ __volatile__("movq $" STRINGIFY(GETPID_SYSCALL) ", %%rax\n"
                                                          "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
#if defined(__APPLE__)
#define EXIT_SYSCALL 0x2000001
#else
#define EXIT_SYSCALL 231
#endif
  uint64_t ret;
  __asm__ __volatile__("movq $" STRINGIFY(EXIT_SYSCALL) ", %%rax\n"
                                                        "syscall\n"
                       : "=a"(ret)
                       : "D"(code)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

#if !defined(__APPLE__)
// We use a stack-allocated buffer for string manipulation in many pieces of
// this code, including the code that prints each line of the fdata file. This
// buffer needs to accomodate large function names, but shouldn't be arbitrarily
// large (dynamically allocated) for simplicity of our memory space usage.

// Declare some syscall wrappers we use throughout this code to avoid linking
// against system libc.
uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  __asm__ __volatile__("movq $2, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(pathname), "S"(flags), "d"(mode)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

long __getdents64(unsigned int fd, dirent64 *dirp, size_t count) {
  long ret;
  __asm__ __volatile__("movq $217, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(fd), "S"(dirp), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  uint64_t ret;
  __asm__ __volatile__("movq $89, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(pathname), "S"(buf), "d"(bufsize)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  __asm__ __volatile__("movq $8, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(pos), "d"(whence)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __ftruncate(uint64_t fd, uint64_t length) {
  int ret;
  __asm__ __volatile__("movq $77, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(length)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __close(uint64_t fd) {
  uint64_t ret;
  __asm__ __volatile__("movq $3, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  __asm__ __volatile__("movq $28, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(length), "d"(advice)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __uname(struct UtsNameTy *Buf) {
  int Ret;
  __asm__ __volatile__("movq $63, %%rax\n"
                       "syscall\n"
                       : "=a"(Ret)
                       : "D"(Buf)
                       : "cc", "rcx", "r11", "memory");
  return Ret;
}

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  __asm__ __volatile__("movq $35, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(req), "S"(rem)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  __asm__ __volatile__("movq $57, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  __asm__ __volatile__("movq $10, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(len), "d"(prot)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  __asm__ __volatile__("movq $110, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __setpgid(uint64_t pid, uint64_t pgid) {
  int ret;
  __asm__ __volatile__("movq $109, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid), "S"(pgid)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getpgid(uint64_t pid) {
  uint64_t ret;
  __asm__ __volatile__("movq $121, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __kill(uint64_t pid, int sig) {
  int ret;
  __asm__ __volatile__("movq $62, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(pid), "S"(sig)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __fsync(int fd) {
  int ret;
  __asm__ __volatile__("movq $74, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

//              %rdi      %rsi         %rdx        %r10         %r8
// sys_prctl  int option  unsigned    unsigned    unsigned    unsigned
//                        long arg2   long arg3   long arg4   long arg5
int __prctl(int Option, unsigned long Arg2, unsigned long Arg3,
            unsigned long Arg4, unsigned long Arg5) {
  int Ret;
  register long rdx asm("rdx") = Arg3;
  register long r8 asm("r8") = Arg5;
  register long r10 asm("r10") = Arg4;
  __asm__ __volatile__("movq $157, %%rax\n"
                       "syscall\n"
                       : "=a"(Ret)
                       : "D"(Option), "S"(Arg2), "d"(rdx), "r"(r10), "r"(r8)
                       :);
  return Ret;
}

#endif

} // anonymous namespace

#endif
