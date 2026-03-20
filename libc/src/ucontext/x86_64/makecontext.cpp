//===-- Implementation of makecontext for x86_64 --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/makecontext.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "include/llvm-libc-types/ucontext_t.h"
#include <stdarg.h>

#include "src/__support/OSUtil/exit.h"
#include "src/ucontext/setcontext.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" void __makecontext_trampoline_c(ucontext_t *uc_link) {
  if (uc_link)
    setcontext(uc_link);

  internal::exit(0);
}

[[gnu::naked]] void __makecontext_trampoline() {
  asm(R"(
      mov %rbx, %rdi
      call __makecontext_trampoline_c
      hlt
  )");
}

LLVM_LIBC_FUNCTION(void, makecontext,
                   (ucontext_t * ucp, void (*func)(void), int argc, ...)) {
  if (!ucp || !func)
    return;

  // System V AMD64 ABI requirements.
  constexpr uintptr_t STACK_ALIGN_BYTES = 16;
  constexpr uintptr_t STACK_ALIGN_MASK = ~(STACK_ALIGN_BYTES - 1);
  constexpr int REGISTER_ARGS_COUNT = 6;
  constexpr uintptr_t ARG_SIZE = sizeof(greg_t);

  uintptr_t stack_top =
      reinterpret_cast<uintptr_t>(ucp->uc_stack.ss_sp) + ucp->uc_stack.ss_size;
  stack_top &= STACK_ALIGN_MASK;

  int stack_args = argc > REGISTER_ARGS_COUNT ? argc - REGISTER_ARGS_COUNT : 0;

  uintptr_t new_rsp = stack_top - stack_args * ARG_SIZE;
  new_rsp &= STACK_ALIGN_MASK;

  // The System V ABI requires the stack to be 16-byte aligned before the 'call'
  // instruction. When a function is entered, the return address has been
  // pushed, making the stack misaligned by 8. We simulate this state by
  // subtracting 8, storing the trampoline address at the top of the stack.
  new_rsp -= ARG_SIZE;

  greg_t *stack_area = reinterpret_cast<greg_t *>(new_rsp);
  stack_area[0] = reinterpret_cast<greg_t>(&__makecontext_trampoline);

  va_list ap;
  va_start(ap, argc);
  if (argc > 0)
    ucp->uc_mcontext.gregs[REG_RDI] = va_arg(ap, greg_t);
  if (argc > 1)
    ucp->uc_mcontext.gregs[REG_RSI] = va_arg(ap, greg_t);
  if (argc > 2)
    ucp->uc_mcontext.gregs[REG_RDX] = va_arg(ap, greg_t);
  if (argc > 3)
    ucp->uc_mcontext.gregs[REG_RCX] = va_arg(ap, greg_t);
  if (argc > 4)
    ucp->uc_mcontext.gregs[REG_R8] = va_arg(ap, greg_t);
  if (argc > 5)
    ucp->uc_mcontext.gregs[REG_R9] = va_arg(ap, greg_t);

  for (int i = 0; i < stack_args; ++i) {
    stack_area[i + 1] = va_arg(ap, greg_t);
  }

  va_end(ap);

  ucp->uc_mcontext.gregs[REG_RIP] = reinterpret_cast<greg_t>(func);
  ucp->uc_mcontext.gregs[REG_RSP] = new_rsp;
  ucp->uc_mcontext.gregs[REG_RBX] = reinterpret_cast<greg_t>(ucp->uc_link);
}

} // namespace LIBC_NAMESPACE_DECL
