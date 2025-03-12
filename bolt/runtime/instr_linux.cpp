//===------------------ bolt/runtime/instr_linux.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BOLT runtime library for intrumenting Linux kernel.
//
//===----------------------------------------------------------------------===//

#include "common.h"

#ifndef __linux__
#error "For Linux only"
#endif

#pragma GCC visibility push(hidden)

extern "C" {

extern void (*__bolt_ind_call_counter_func_pointer)();
extern void (*__bolt_ind_tailcall_counter_func_pointer)();

}

namespace {

// Base address which we substract from recorded PC values when searching for
// indirect call description entries. Needed because indCall descriptions are
// mapped read-only and contain static addresses. Initialized in
// __bolt_instr_setup.
uint64_t TextBaseAddress = 0;

} // anonymous namespace

extern "C" void __bolt_instr_indirect_call();
extern "C" void __bolt_instr_indirect_tailcall();

extern "C" __attribute((force_align_arg_pointer)) void
instrumentIndirectCall(uint64_t Target, uint64_t IndCallID) {}

/// We receive as in-stack arguments the identifier of the indirect call site
/// as well as the target address for the call
extern "C" __attribute((naked)) void __bolt_instr_indirect_call() {
#if defined(__aarch64__)
  // clang-format off
  __asm__ __volatile__(SAVE_ALL
                       "ldp x0, x1, [sp, #288]\n"
                       "bl instrumentIndirectCall\n"
                       RESTORE_ALL
                       "ret\n"
                       :::);
  // clang-format on
#else
  // clang-format off
  __asm__ __volatile__(SAVE_ALL
                       "mov 0xa0(%%rsp), %%rdi\n"
                       "mov 0x98(%%rsp), %%rsi\n"
                       "call instrumentIndirectCall\n"
                       RESTORE_ALL
                       "ret\n"
                       :::);
  // clang-format on
#endif
}

extern "C" __attribute((naked)) void __bolt_instr_indirect_tailcall() {
#if defined(__aarch64__)
  // clang-format off
  __asm__ __volatile__(SAVE_ALL
                       "ldp x0, x1, [sp, #288]\n"
                       "bl instrumentIndirectCall\n"
                       RESTORE_ALL
                       "ret\n"
                       :::);
  // clang-format on
#else
  // clang-format off
  __asm__ __volatile__(SAVE_ALL
                       "mov 0x98(%%rsp), %%rdi\n"
                       "mov 0x90(%%rsp), %%rsi\n"
                       "call instrumentIndirectCall\n"
                       RESTORE_ALL
                       "ret\n"
                       :::);
  // clang-format on
#endif
}

extern "C" void __attribute((force_align_arg_pointer)) __bolt_instr_setup() {
  __bolt_ind_call_counter_func_pointer = __bolt_instr_indirect_call;
  __bolt_ind_tailcall_counter_func_pointer = __bolt_instr_indirect_tailcall;
  TextBaseAddress = getTextBaseAddress();
}
