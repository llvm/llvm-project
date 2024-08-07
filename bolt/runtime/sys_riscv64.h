#ifndef LLVM_TOOLS_LLVM_BOLT_SYS_RISCV64
#define LLVM_TOOLS_LLVM_BOLT_SYS_RISCV64

#include <dirent.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>

#define STRINGIFY(X) #X
#define XSTRINGIFY(X) STRINGIFY(X)

// Syscalls are wrapped as follows: we declare a C function with the same type
// as the syscall. This function is then implemented in global assembly by
// simply loading the syscall number and calling ecall. This way, the compiler
// will correctly prepare all argument registers when calling the wrapper. This
// works because the syscall calling convention matches the user space one.
// clang-format off
#define WRAP_SYSCALL(Name, ReturnType, ...) \
  ReturnType __##Name(__VA_ARGS__); \
  asm( \
    XSTRINGIFY(__##Name) ":\n\t" \
    "li a7, " XSTRINGIFY(SYS_##Name) "\n\t" \
    "ecall\n\t" \
    "ret\n\t" \
  )
// clang-format on

extern "C" {

WRAP_SYSCALL(read, ssize_t, int, void *, size_t);
WRAP_SYSCALL(write, ssize_t, int, const void *, size_t);
WRAP_SYSCALL(mmap, void *, size_t, size_t, int, int, int, off_t);
WRAP_SYSCALL(munmap, int, void *, size_t);
WRAP_SYSCALL(exit, void, int);
WRAP_SYSCALL(openat, int, int, const char *, int, mode_t);
WRAP_SYSCALL(readlinkat, ssize_t, int, const char *, char *, size_t);
WRAP_SYSCALL(getdents64, long, unsigned int, struct dirent64 *, size_t);
WRAP_SYSCALL(lseek, off_t, int, off_t, int);
WRAP_SYSCALL(fsync, int, int);
WRAP_SYSCALL(ftruncate, int, int, off_t);
WRAP_SYSCALL(close, int, int);
WRAP_SYSCALL(setpgid, int, pid_t, pid_t);
WRAP_SYSCALL(getpgid, pid_t, pid_t);
WRAP_SYSCALL(getpid, pid_t, );
WRAP_SYSCALL(getppid, pid_t, );
WRAP_SYSCALL(nanosleep, int, struct timespec *, struct timespec *);
WRAP_SYSCALL(rt_sigprocmask, int, int, const void *, void *, size_t);
WRAP_SYSCALL(kill, int, pid_t, int);
WRAP_SYSCALL(clone, long, unsigned long, unsigned long, int *, unsigned long,
             int *);

pid_t __fork() {
  return __clone(CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID | SIGCHLD, 0, 0, 0,
                 0);
}

int __open(const char *pathname, int flags, mode_t mode) {
  return __openat(AT_FDCWD, pathname, flags, mode);
}

ssize_t __readlink(const char *pathname, char *buf, size_t bufsize) {
  return __readlinkat(AT_FDCWD, pathname, buf, bufsize);
}

int __sigprocmask(int how, const void *set, void *oldset) {
  return __rt_sigprocmask(how, set, oldset, 8);
}
}

// Anonymous namespace covering everything but our library entry point
namespace {

// Get the difference between runtime address of .text section and static
// address in section header table. Can be extracted from arbitrary pc value
// recorded at runtime to get the corresponding static address, which in turn
// can be used to search for indirect call description. Needed because indirect
// call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("lla %0, __hot_end\n\t"
                   "lui %1, %%hi(__hot_end)\n\t"
                   "addi %1, %1, %%lo(__hot_end)\n\t"
                   : "=r"(DynAddr), "=r"(StaticAddr));
  return DynAddr - StaticAddr;
}

} // anonymous namespace

#endif
