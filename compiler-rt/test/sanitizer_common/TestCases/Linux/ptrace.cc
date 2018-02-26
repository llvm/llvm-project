// RUN: %clangxx -O0 %s -o %t && %run %t

// UNSUPPORTED: android

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <unistd.h>
#include <elf.h>
#if __mips64 || __arm__
 #include <asm/ptrace.h>
 #include <sys/procfs.h>
#endif
#ifdef __aarch64__
// GLIBC 2.20+ sys/user does not include asm/ptrace.h
 #include <asm/ptrace.h>
#endif

int main(void) {
  pid_t pid;
  pid = fork();
  if (pid == 0) { // child
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("/bin/true", "true", NULL);
  } else {
    wait(NULL);
    int res;

#if __x86_64__
    user_regs_struct regs;
    res = ptrace(PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
    if (regs.rip)
      printf("%zx\n", regs.rip);

    user_fpregs_struct fpregs;
    res = ptrace(PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if (fpregs.mxcsr)
      printf("%x\n", fpregs.mxcsr);
#endif // __x86_64__

#if (__powerpc64__ || __mips64 || __arm__)
    struct pt_regs regs;
    res = ptrace((enum __ptrace_request)PTRACE_GETREGS, pid, NULL, &regs);
    assert(!res);
#if (__powerpc64__)
    if (regs.nip)
      printf("%lx\n", regs.nip);
#elif (__mips64)
    if (regs.cp0_epc)
    printf("%lx\n", regs.cp0_epc);
#elif (__arm__)
    if (regs.ARM_pc)
    printf("%lx\n", regs.ARM_pc);
#endif
#if (__powerpc64 || __mips64)
    elf_fpregset_t fpregs;
    res = ptrace((enum __ptrace_request)PTRACE_GETFPREGS, pid, NULL, &fpregs);
    assert(!res);
    if ((elf_greg_t)fpregs[32]) // fpscr
      printf("%lx\n", (elf_greg_t)fpregs[32]);
#elif (__arm__)
    char regbuf[ARM_VFPREGS_SIZE];
    res = ptrace((enum __ptrace_request)PTRACE_GETVFPREGS, pid, 0, regbuf);
    assert(!res);
    unsigned fpscr = *(unsigned*)(regbuf + (32 * 8));
    printf ("%x\n", fpscr);
#endif
#endif // (__powerpc64__ || __mips64 || __arm__)

#if (__aarch64__)
    struct iovec regset_io;

    struct user_pt_regs regs;
    regset_io.iov_base = &regs;
    regset_io.iov_len = sizeof(regs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, (void*)&regset_io);
    assert(!res);
    if (regs.pc)
      printf("%llx\n", regs.pc);

    struct user_fpsimd_state fpregs;
    regset_io.iov_base = &fpregs;
    regset_io.iov_len = sizeof(fpregs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_FPREGSET, (void*)&regset_io);
    assert(!res);
    if (fpregs.fpsr)
      printf("%x\n", fpregs.fpsr);
#endif // (__aarch64__)

#if (__s390__)
    struct iovec regset_io;

    struct _user_regs_struct regs;
    regset_io.iov_base = &regs;
    regset_io.iov_len = sizeof(regs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_PRSTATUS, (void*)&regset_io);
    assert(!res);
    if (regs.psw.addr)
      printf("%lx\n", regs.psw.addr);

    struct _user_fpregs_struct fpregs;
    regset_io.iov_base = &fpregs;
    regset_io.iov_len = sizeof(fpregs);
    res = ptrace(PTRACE_GETREGSET, pid, (void*)NT_FPREGSET, (void*)&regset_io);
    assert(!res);
    if (fpregs.fpc)
      printf("%x\n", fpregs.fpc);
#endif // (__s390__)

    siginfo_t siginfo;
    res = ptrace(PTRACE_GETSIGINFO, pid, NULL, &siginfo);
    assert(!res);
    assert(siginfo.si_pid == pid);

    ptrace(PTRACE_CONT, pid, NULL, NULL);

    wait(NULL);
  }
  return 0;
}
