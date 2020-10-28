// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
// XFAIL: mips
//
// RUN: %clangxx_asan -O0 %s -o %t && %run %t
// RUN: %clangxx_asan -DPOSITIVE -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/uio.h> // for iovec
#include <elf.h> // for NT_PRSTATUS
#ifdef __aarch64__
# include <asm/ptrace.h>
#endif

#if defined(__i386__) || defined(__x86_64__)
typedef user_regs_struct   regs_struct;
typedef user_fpregs_struct fpregs_struct;
#if defined(__i386__)
#define REG_IP  eip
#else
#define REG_IP  rip
#endif
#define PRINT_REG_PC(__regs)    printf ("%lx\n", (unsigned long) (__regs.REG_IP))
#define PRINT_REG_FP(__fpregs)  printf ("%lx\n", (unsigned long) (__fpregs.cwd))
#define __PTRACE_FPREQUEST PTRACE_GETFPREGS

#elif defined(__aarch64__)
typedef struct user_pt_regs      regs_struct;
typedef struct user_fpsimd_state fpregs_struct;
#define PRINT_REG_PC(__regs)    printf ("%x\n", (unsigned) (__regs.pc))
#define PRINT_REG_FP(__fpregs)  printf ("%x\n", (unsigned) (__fpregs.fpsr))
#define ARCH_IOVEC_FOR_GETREGSET

#elif defined(__powerpc64__)
typedef struct pt_regs regs_struct;
typedef elf_fpregset_t fpregs_struct;
#define PRINT_REG_PC(__regs)    printf ("%lx\n", (unsigned long) (__regs.nip))
#define PRINT_REG_FP(__fpregs)  printf ("%lx\n", (elf_greg_t)fpregs[32])
#define ARCH_IOVEC_FOR_GETREGSET

#elif defined(__mips__)
typedef struct pt_regs regs_struct;
typedef elf_fpregset_t fpregs_struct;
#define PRINT_REG_PC(__regs)    printf ("%lx\n", (unsigned long) (__regs.cp0_epc))
#define PRINT_REG_FP(__fpregs)  printf ("%lx\n", (elf_greg_t) (__fpregs[32]))
#define __PTRACE_FPREQUEST PTRACE_GETFPREGS

#elif defined(__arm__)
# include <asm/ptrace.h>
# include <sys/procfs.h>
typedef struct pt_regs regs_struct;
typedef char fpregs_struct[ARM_VFPREGS_SIZE];
#define PRINT_REG_PC(__regs)    printf ("%x\n", (unsigned) (__regs.ARM_pc))
#define PRINT_REG_FP(__fpregs)  printf ("%x\n", (unsigned) (__fpregs + 32 * 8))
#define __PTRACE_FPREQUEST PTRACE_GETVFPREGS

#elif defined(__s390__)
typedef _user_regs_struct   regs_struct;
typedef _user_fpregs_struct fpregs_struct;
#define PRINT_REG_PC(__regs)    printf ("%lx\n", (unsigned long) (__regs.psw.addr))
#define PRINT_REG_FP(__fpregs)  printf ("%lx\n", (unsigned long) (__fpregs.fpc))
#define ARCH_IOVEC_FOR_GETREGSET

#elif defined(__riscv) && (__riscv_xlen == 64)
#include <asm/ptrace.h>
typedef user_regs_struct regs_struct;
typedef __riscv_q_ext_state fpregs_struct;
#define PRINT_REG_PC(__regs) printf("%lx\n", (unsigned long)(__regs.pc))
#define PRINT_REG_FP(__fpregs) printf("%lx\n", (unsigned long)(__fpregs.fcsr))
#define ARCH_IOVEC_FOR_GETREGSET
#endif


int main(void) {
  pid_t pid;
  pid = fork();
  if (pid == 0) { // child
    ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    execl("/bin/true", "true", NULL);
  } else {
    wait(NULL);
    regs_struct regs;
    regs_struct* volatile pregs = &regs;
#ifdef ARCH_IOVEC_FOR_GETREGSET
    struct iovec regset_io;
#endif
    int res;

#ifdef POSITIVE
    ++pregs;
#endif

#ifdef ARCH_IOVEC_FOR_GETREGSET
# define __PTRACE_REQUEST  PTRACE_GETREGSET
# define __PTRACE_ARGS     (void*)NT_PRSTATUS, (void*)&regset_io
    regset_io.iov_base = pregs;
    regset_io.iov_len = sizeof(regs_struct);
#else
# define __PTRACE_REQUEST  PTRACE_GETREGS
# define __PTRACE_ARGS     NULL, pregs
#endif
    res = ptrace((enum __ptrace_request)__PTRACE_REQUEST, pid, __PTRACE_ARGS);
    // CHECK: AddressSanitizer: stack-buffer-overflow
    // CHECK: {{.*ptrace.cpp:}}[[@LINE-2]]
    assert(!res);
    PRINT_REG_PC(regs);

    fpregs_struct fpregs;
#ifdef ARCH_IOVEC_FOR_GETREGSET
# define __PTRACE_FPREQUEST  PTRACE_GETREGSET
# define __PTRACE_FPARGS     (void*)NT_PRSTATUS, (void*)&regset_io
    regset_io.iov_base = &fpregs;
    regset_io.iov_len = sizeof(fpregs_struct);
    res = ptrace((enum __ptrace_request)PTRACE_GETREGSET, pid, (void*)NT_FPREGSET,
                 (void*)&regset_io);
#else
# define __PTRACE_FPARGS     NULL, &fpregs
#endif
    res = ptrace((enum __ptrace_request)__PTRACE_FPREQUEST, pid, __PTRACE_FPARGS);
    assert(!res);
    PRINT_REG_FP(fpregs);

#ifdef __i386__
    user_fpxregs_struct fpxregs;
    res = ptrace(PTRACE_GETFPXREGS, pid, NULL, &fpxregs);
    assert(!res);
    printf("%lx\n", (unsigned long)fpxregs.mxcsr);
#endif

    ptrace(PTRACE_CONT, pid, NULL, NULL);
    wait(NULL);
  }
  return 0;
}
