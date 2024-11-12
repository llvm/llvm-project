//===-- Common macros for jmpbuf  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBC_SRC_SETJMP_X86_64_COMMON_H
#define LIBC_SRC_SETJMP_X86_64_COMMON_H

#include "include/llvm-libc-macros/offsetof-macro.h"

//===----------------------------------------------------------------------===//
// Architecture specific macros for x86_64.
//===----------------------------------------------------------------------===//

#ifdef __i386__
#define RET_REG eax
#define BASE_REG ecx
#define MUL_REG edx
#define STACK_REG esp
#define PC_REG eip
#define NORMAL_STORE_REGS ebx, esi, edi, ebp
#define STORE_ALL_REGS(M) M(ebx) M(esi) M(edi) M(ebp)
#define LOAD_ALL_REGS(M) M(ebx) M(esi) M(edi) M(ebp) M(esp)
#define DECLARE_ALL_REGS(M) M(ebx), M(esi), M(edi), M(ebp), M(esp), M(eip)
#define LOAD_BASE() "mov 4(%%esp), %%ecx\n\t"
#define CALCULATE_RETURN_VALUE()                                               \
  "mov 0x8(%%esp), %%eax"                                                      \
  "cmp $0x1, %%eax\n\t"                                                        \
  "adc $0x0, %%eax\n\t"
#else
#define RET_REG rax
#define BASE_REG rdi
#define MUL_REG rdx
#define STACK_REG rsp
#define PC_REG rip
#define STORE_ALL_REGS(M) M(rbx) M(rbp) M(r12) M(r13) M(r14) M(r15)
#define LOAD_ALL_REGS(M) M(rbx) M(rbp) M(r12) M(r13) M(r14) M(r15) M(rsp)
#define DECLARE_ALL_REGS(M)                                                    \
  M(rbx), M(rbp), M(r12), M(r13), M(r14), M(r15), M(rsp), M(rip)
#define LOAD_BASE()
#define CALCULATE_RETURN_VALUE()                                               \
  "cmp $0x1, %%esi\n\t"                                                        \
  "adc $0x0, %%esi\n\t"                                                        \
  "mov %%rsi, %%rax\n\t"
#endif

//===----------------------------------------------------------------------===//
// Utility macros.
//===----------------------------------------------------------------------===//

#define _STR(X) #X
#define STR(X) _STR(X)
#define REG(X) "%%" STR(X)
#define XOR(X, Y) "xor " REG(X) ", " REG(Y) "\n\t"
#define MOV(X, Y) "mov " REG(X) ", " REG(Y) "\n\t"
#define STORE(R, OFFSET, BASE)                                                 \
  "mov " REG(R) ", %c[" STR(OFFSET) "](" REG(BASE) ")\n\t"
#define LOAD(OFFSET, BASE, R)                                                  \
  "mov %c[" STR(OFFSET) "](" REG(BASE) "), " REG(R) "\n\t"
#define COMPUTE_STACK_TO_RET()                                                 \
  "lea " STR(__SIZEOF_POINTER__) "(" REG(STACK_REG) "), " REG(RET_REG) "\n\t"
#define COMPUTE_PC_TO_RET() "mov (" REG(STACK_REG) "), " REG(RET_REG) "\n\t"
#define RETURN() "ret\n\t"
#define DECLARE_OFFSET(X) [X] "i"(offsetof(__jmp_buf, X))
#define CMP_MEM_REG(OFFSET, BASE, DST)                                         \
  "cmp %c[" STR(OFFSET) "](" REG(BASE) "), " REG(DST) "\n\t"
#define JNE_LABEL(LABEL) "jne " STR(LABEL) "\n\t"

//===----------------------------------------------------------------------===//
// Checksum related macros.
//===----------------------------------------------------------------------===//
// For now, the checksum is computed with a simple multiply-xor-rotation
// algorithm. The pesudo code is as follows:
//
// def checksum(x, acc):
//     masked = x ^ MASK
//     high, low = full_multiply(masked, acc)
//     return rotate(high ^ low, ROTATION)
//
// Similar other multiplication-based hashing, zero inputs
// for the `full_multiply` function may pollute the checksum with zero.
// However, user inputs are always masked where the initial ACC amd MASK are
// generated with random entropy and ROTATION is a fixed prime number. It should
// be of a ultra-low chance for masked or acc being zero given a good quality of
// system-level entropy.
//
// Notice that on x86-64, one-operand form of `mul` instruction:
//  mul %rdx
// has the following effect:
//  RAX = LOW(RDX * RAX)
//  RDX = HIGH(RDX * RAX)
//===----------------------------------------------------------------------===//

#if LIBC_COPT_SETJMP_FORTIFICATION
#define XOR_MASK(X) "xor %[value_mask], " REG(X) "\n\t"
#define MUL(X) "mul " REG(X) "\n\t"
#define ROTATE(X) "rol $%c[rotation], " REG(X) "\n\t"
#define ACCUMULATE_CHECKSUM() MUL(MUL_REG) XOR(RET_REG, MUL_REG) ROTATE(MUL_REG)

#define LOAD_CHKSUM_STATE_REGS() "mov %[checksum_cookie], " REG(MUL_REG) "\n\t"

#define STORE_REG(SRC)                                                         \
  MOV(SRC, RET_REG) XOR_MASK(RET_REG) STORE(RET_REG, SRC, BASE_REG)
#define STORE_STACK()                                                          \
  COMPUTE_STACK_TO_RET()                                                       \
  XOR_MASK(RET_REG)                                                            \
  STORE(RET_REG, STACK_REG, BASE_REG)

#define STORE_PC()                                                             \
  COMPUTE_PC_TO_RET()                                                          \
  XOR_MASK(RET_REG)                                                            \
  STORE(RET_REG, PC_REG, BASE_REG)

#define STORE_CHECKSUM() STORE(MUL_REG, __chksum, BASE_REG)
#define EXAMINE_CHECKSUM()                                                     \
  LOAD(PC_REG, BASE_REG, RET_REG)                                              \
  ACCUMULATE_CHECKSUM()                                                        \
  CMP_MEM_REG(__chksum, BASE_REG, MUL_REG)                                     \
  JNE_LABEL(__libc_jmpbuf_corruption)

#define RESTORE_PC()                                                           \
  LOAD(PC_REG, BASE_REG, BASE_REG)                                             \
  XOR_MASK(BASE_REG)                                                           \
  "jmp *" REG(BASE_REG)
#define RESTORE_REG(SRC)                                                       \
  LOAD(SRC, BASE_REG, RET_REG)                                                 \
  MOV(RET_REG, SRC)                                                            \
  ACCUMULATE_CHECKSUM() XOR_MASK(SRC)
#else
#define XOR_MASK(X)
#define ACCUMULATE_CHECKSUM()
#define LOAD_CHKSUM_STATE_REGS()
#define STORE_REG(SRC) STORE(SRC, SRC, BASE_REG)
#define STORE_STACK() COMPUTE_STACK_TO_RET() STORE(RET_REG, STACK_REG, BASE_REG)
#define STORE_PC() COMPUTE_PC_TO_RET() STORE(RET_REG, PC_REG, BASE_REG)
#define STORE_CHECKSUM()
#define EXAMINE_CHECKSUM()
#define RESTORE_PC() "jmp *%c[" STR(PC_REG) "](" REG(BASE_REG) ")\n\t"
#define RESTORE_REG(SRC) LOAD(SRC, BASE_REG, SRC)
#endif

#define STORE_REG_ACCUMULATE(SRC) STORE_REG(SRC) ACCUMULATE_CHECKSUM()

#endif // LIBC_SRC_SETJMP_X86_64_COMMON_H
