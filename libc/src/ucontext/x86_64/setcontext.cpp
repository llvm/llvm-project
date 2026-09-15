//===-- Implementation of setcontext for x86_64 ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/setcontext.h"
#include "include/llvm-libc-types/ucontext_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"
#include "include/llvm-libc-macros/signal-macros.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

__attribute__((naked)) LLVM_LIBC_FUNCTION(int, setcontext,
                                          (const ucontext_t *ucp)) noexcept {
  asm(R"(
      # ucp is in rdi
      
      # Restore the signal mask using rt_sigprocmask syscall.
      # rt_sigprocmask(SIG_SETMASK, &ucp->uc_sigmask, NULL, sizeof(sigset_t))
      # Note: Restoring the signal mask early means that if a signal
      # arrives before the context switch is complete, it will run on
      # the old stack with the new mask. Doing this later is difficult
      # because the syscall clobbers registers.
      #
      # Note: We could avoid these stack operations by saving rdi in a
      # non-volatile register (like r12) across the syscall, since all
      # registers will be overwritten anyway. We stick to the stack for
      # simplicity and readability.
      pushq %%rdi # Save ucp
      leaq %c[sigmask](%%rdi), %%rsi # set = &ucp->uc_sigmask
      xorq %%rdx, %%rdx # oldset = NULL
      movq $%c[sigset_size], %%r10 # sigsetsize = sizeof(sigset_t)
      movq $%c[sig_setmask], %%rdi # how = SIG_SETMASK
      movq $%c[syscall_num], %%rax
      syscall
      popq %%rdi # Restore ucp

      # Restore floating point state
      fxrstorq %c[fpregs_mem](%%rdi)

      # Restore other general purpose registers
      mov %c[r8](%%rdi), %%r8
      mov %c[r9](%%rdi), %%r9
      mov %c[r10](%%rdi), %%r10
      mov %c[r11](%%rdi), %%r11
      mov %c[r12](%%rdi), %%r12
      mov %c[r13](%%rdi), %%r13
      mov %c[r14](%%rdi), %%r14
      mov %c[r15](%%rdi), %%r15
      mov %c[rbp](%%rdi), %%rbp
      mov %c[rbx](%%rdi), %%rbx
      mov %c[rdx](%%rdi), %%rdx
      mov %c[rax](%%rdi), %%rax
      mov %c[rcx](%%rdi), %%rcx

      # Restore stack pointer
      mov %c[rsp](%%rdi), %%rsp
      # Push saved RIP onto the new stack to use ret later
      pushq %c[rip](%%rdi)
      
      # Restore RSI and RDI last
      mov %c[rsi](%%rdi), %%rsi
      mov %c[rdi](%%rdi), %%rdi

      retq
      )" ::[sigset_size] "i"(sizeof(sigset_t)),
      [syscall_num] "i"(SYS_rt_sigprocmask), [sig_setmask] "i"(SIG_SETMASK),
      [r8] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R8])),
      [r9] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R9])),
      [r10] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R10])),
      [r11] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R11])),
      [r12] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R12])),
      [r13] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R13])),
      [r14] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R14])),
      [r15] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_R15])),
      [rdi] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RDI])),
      [rsi] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RSI])),
      [rbp] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RBP])),
      [rbx] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RBX])),
      [rdx] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RDX])),
      [rax] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RAX])),
      [rcx] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RCX])),
      [rsp] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RSP])),
      [rip] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.gregs[REG_RIP])),
      [fpregs_mem] "i"(__builtin_offsetof(ucontext_t, __fpregs_mem)),
      [sigmask] "i"(__builtin_offsetof(ucontext_t, uc_sigmask))
      : "memory", "rcx", "r11");
}

} // namespace LIBC_NAMESPACE_DECL
