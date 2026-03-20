//===-- Implementation of swapcontext for x86_64 --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/swapcontext.h"
#include "include/llvm-libc-types/ucontext_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, swapcontext,
                   (ucontext_t * oucp, const ucontext_t *ucp)) {
  asm(R"(
      # oucp is in rdi, ucp is in rsi

      // Save current context into oucp
      // Save general purpose registers
      mov %%r8, %c[r8](%%rdi)
      mov %%r9, %c[r9](%%rdi)
      mov %%r10, %c[r10](%%rdi)
      mov %%r11, %c[r11](%%rdi)
      mov %%r12, %c[r12](%%rdi)
      mov %%r13, %c[r13](%%rdi)
      mov %%r14, %c[r14](%%rdi)
      mov %%r15, %c[r15](%%rdi)
      mov %%rdi, %c[rdi](%%rdi) # oucp itself
      mov %%rsi, %c[rsi](%%rdi) # ucp
      mov %%rbp, %c[rbp](%%rdi)
      mov %%rbx, %c[rbx](%%rdi)
      mov %%rdx, %c[rdx](%%rdi)
      // setcontext should return 0 when resumed by setcontext.
      // So we save 0 into the RAX register of the context.
      movq $0, %c[rax](%%rdi)
      mov %%rcx, %c[rcx](%%rdi)

      // The stack pointer before the call is rsp + sizeof(void*).
      // The return address was pushed when this function was called.
      // Save instruction pointer and stack pointer
      mov (%%rsp), %%rax
      mov %%rax, %c[rip](%%rdi)
      lea %c[ret_size](%%rsp), %%rax
      mov %%rax, %c[rsp](%%rdi)

      // Save floating point state
      fxsaveq %c[fpregs_mem](%%rdi)
      // Point mcontext.fpregs to our internal FP storage
      lea %c[fpregs_mem](%%rdi), %%rax
      mov %%rax, %c[fpregs_ptr](%%rdi)

      // Capture oucp signal mask and restore ucp signal mask atomically.
      // rt_sigprocmask(SIG_SETMASK, &ucp->uc_sigmask, &oucp->uc_sigmask, sizeof(sigset_t))
      // oucp is in rdi, ucp is in rsi
      pushq %%rdi # Save oucp
      pushq %%rsi # Save ucp
      leaq %c[sigmask](%%rdi), %%rdx # oldset = &oucp->uc_sigmask
      leaq %c[sigmask](%%rsi), %%rsi # set = &ucp->uc_sigmask
      movq $%c[sigset_size], %%r10 # sigsetsize = sizeof(sigset_t)
      movq $2, %%rdi # how = SIG_SETMASK
      movq $%c[syscall_num], %%rax
      syscall
      popq %%rsi # Restore ucp (new context)
      popq %%rdi # Restore oucp (old context - not needed but for clean stack)

      // Restore context from ucp (now in rsi)
      // Restore floating point state
      fxrstorq %c[fpregs_mem](%%rsi)

      // Restore general purpose registers EXECPT rdi, rsi, rsp, rip
      mov %c[r8](%%rsi), %%r8
      mov %c[r9](%%rsi), %%r9
      mov %c[r10](%%rsi), %%r10
      mov %c[r11](%%rsi), %%r11
      mov %c[r12](%%rsi), %%r12
      mov %c[r13](%%rsi), %%r13
      mov %c[r14](%%rsi), %%r14
      mov %c[r15](%%rsi), %%r15
      mov %c[rbp](%%rsi), %%rbp
      mov %c[rbx](%%rsi), %%rbx
      mov %c[rdx](%%rsi), %%rdx
      mov %c[rax](%%rsi), %%rax
      mov %c[rcx](%%rsi), %%rcx

      // Restore stack pointer and instruction pointer
      mov %c[rsp](%%rsi), %%rsp
      mov %c[rip](%%rsi), %%r11 # Use r11 as temp for rip
      
      // Restore RSI and RDI last
      mov %c[rdi](%%rsi), %%rdi
      mov %c[rsi](%%rsi), %%rsi

      jmpq *%%r11 # Jump to the saved instruction pointer
      )" ::[ret_size] "i"(sizeof(void *)),
      [sigset_size] "i"(sizeof(sigset_t)),
      [syscall_num] "i"(SYS_rt_sigprocmask),
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
      [fpregs_ptr] "i"(__builtin_offsetof(ucontext_t, uc_mcontext.fpregs)),
      [sigmask] "i"(__builtin_offsetof(ucontext_t, uc_sigmask))
      : "memory", "rcx", "r11");
}

} // namespace LIBC_NAMESPACE_DECL
