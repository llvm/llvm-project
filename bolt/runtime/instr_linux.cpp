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

#ifndef __linux__
#error "For Linux only"
#endif

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "stp x0, x1, [sp, #-16]!\n"                                                  \
  "stp x2, x3, [sp, #-16]!\n"                                                  \
  "stp x4, x5, [sp, #-16]!\n"                                                  \
  "stp x6, x7, [sp, #-16]!\n"                                                  \
  "stp x8, x9, [sp, #-16]!\n"                                                  \
  "stp x10, x11, [sp, #-16]!\n"                                                \
  "stp x12, x13, [sp, #-16]!\n"                                                \
  "stp x14, x15, [sp, #-16]!\n"                                                \
  "stp x16, x17, [sp, #-16]!\n"                                                \
  "stp x18, x19, [sp, #-16]!\n"                                                \
  "stp x20, x21, [sp, #-16]!\n"                                                \
  "stp x22, x23, [sp, #-16]!\n"                                                \
  "stp x24, x25, [sp, #-16]!\n"                                                \
  "stp x26, x27, [sp, #-16]!\n"                                                \
  "stp x28, x29, [sp, #-16]!\n"                                                \
  "str x30, [sp,#-16]!\n"
// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "ldr x30, [sp], #16\n"                                                       \
  "ldp x28, x29, [sp], #16\n"                                                  \
  "ldp x26, x27, [sp], #16\n"                                                  \
  "ldp x24, x25, [sp], #16\n"                                                  \
  "ldp x22, x23, [sp], #16\n"                                                  \
  "ldp x20, x21, [sp], #16\n"                                                  \
  "ldp x18, x19, [sp], #16\n"                                                  \
  "ldp x16, x17, [sp], #16\n"                                                  \
  "ldp x14, x15, [sp], #16\n"                                                  \
  "ldp x12, x13, [sp], #16\n"                                                  \
  "ldp x10, x11, [sp], #16\n"                                                  \
  "ldp x8, x9, [sp], #16\n"                                                    \
  "ldp x6, x7, [sp], #16\n"                                                    \
  "ldp x4, x5, [sp], #16\n"                                                    \
  "ldp x2, x3, [sp], #16\n"                                                    \
  "ldp x0, x1, [sp], #16\n"

namespace {

// Get the difference between runtime addrress of .text section and
// static address in section header table. Can be extracted from arbitrary
// pc value recorded at runtime to get the corresponding static address, which
// in turn can be used to search for indirect call description. Needed because
// indirect call descriptions are read-only non-relocatable data.
uint64_t getTextBaseAddress() {
  uint64_t DynAddr;
  uint64_t StaticAddr;
  __asm__ volatile("b .instr%=\n\t"
                   ".StaticAddr%=:\n\t"
                   ".dword __hot_end\n\t"
                   ".instr%=:\n\t"
                   "ldr %0, .StaticAddr%=\n\t"
                   "adrp %1, __hot_end\n\t"
                   "add %1, %1, :lo12:__hot_end\n\t"
                   : "=r"(StaticAddr), "=r"(DynAddr));
  return DynAddr - StaticAddr;
}

} // namespace

#elif defined(__x86_64__)

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

} // namespace

#else
#error "Unsupported architecture"
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
