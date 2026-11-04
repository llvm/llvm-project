//===-- Implementation of getcontext for x86_64 ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/getcontext.h"
#include "include/llvm-libc-types/ucontext_t.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/types/size_t.h"
#include "include/llvm-libc-macros/signal-macros.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

// We use naked because we need to capture the exact register state
// at the moment of the function call, avoiding any compiler prologue/epilogue.
__attribute__((naked)) LLVM_LIBC_FUNCTION(int, getcontext,
                                          (ucontext_t * ucp)) noexcept {
  asm(R"(
      # ucp is in rdi
      
      # Save general purpose registers
      mov %%r8, %c[r8](%%rdi)
      mov %%r9, %c[r9](%%rdi)
      mov %%r10, %c[r10](%%rdi)
      mov %%r11, %c[r11](%%rdi)
      mov %%r12, %c[r12](%%rdi)
      mov %%r13, %c[r13](%%rdi)
      mov %%r14, %c[r14](%%rdi)
      mov %%r15, %c[r15](%%rdi)
      mov %%rdi, %c[rdi](%%rdi)
      mov %%rsi, %c[rsi](%%rdi)
      mov %%rbp, %c[rbp](%%rdi)
      mov %%rbx, %c[rbx](%%rdi)
      mov %%rdx, %c[rdx](%%rdi)
      # getcontext should return 0 when resumed by setcontext.
      # So we save 0 into the RAX register of the context.
      movq $0, %c[rax](%%rdi)
      mov %%rcx, %c[rcx](%%rdi)

      # The stack pointer before the call is rsp + sizeof(void*).
      # The return address was pushed when this function was called.
      # Save instruction pointer and stack pointer
      mov (%%rsp), %%rax
      mov %%rax, %c[rip](%%rdi)
      lea %c[ret_size](%%rsp), %%rax
      mov %%rax, %c[rsp](%%rdi)

      # Save floating point state
      fxsaveq %c[fpregs_mem](%%rdi)
      # Point mcontext.fpregs to our internal FP storage
      lea %c[fpregs_mem](%%rdi), %%rax
      mov %%rax, %c[fpregs_ptr](%%rdi)

      # Capture the signal mask using rt_sigprocmask syscall.
      # rt_sigprocmask(SIG_BLOCK, NULL, &ucp->uc_sigmask, sizeof(sigset_t))
      leaq %c[sigmask](%%rdi), %%rdx # oldset = &ucp->uc_sigmask
      xorq %%rsi, %%rsi # set = NULL
      movq $%c[sig_block], %%rdi # SIG_BLOCK (captured mask in oldset)
      movq $%c[sigset_size], %%r10
      movq $%c[syscall_num], %%rax
      syscall

      # getcontext should return 0 on success
      xor %%eax, %%eax

      retq
      )" ::[ret_size] "i"(sizeof(void *)),
      [sigset_size] "i"(sizeof(sigset_t)),
      [syscall_num] "i"(SYS_rt_sigprocmask), [sig_block] "i"(SIG_BLOCK),
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
      : "memory", "rcx", "r11", "rdi", "rsi", "rax", "r10");
}

} // namespace LIBC_NAMESPACE_DECL
