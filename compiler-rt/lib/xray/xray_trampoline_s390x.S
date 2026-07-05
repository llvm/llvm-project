//===-- xray_trampoline_s390x.s ---------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the s390x-specific assembler for the trampolines.
// 2 versions of the functions are provided: one which does not store the
// vector registers, and one which does store them. The compiler decides
// which to call based on the availability of the vector extension.
//
//===----------------------------------------------------------------------===//

    .text

// Minimal stack frame size
#define STACKSZ  160

// Minimal stack frame size (160) plus space for 8 vector registers a 16 bytes.
#define STACKSZ_VEC  288

//===----------------------------------------------------------------------===//

    .globl  __xray_FunctionEntry
    .p2align    4
    .type   __xray_FunctionEntry,@function
__xray_FunctionEntry:
    # The registers r2-15 of the instrumented function are already saved in the
    # stack frame. On entry, r2 contains the function id, and %r14 the address
    # of the first instruction of the instrumented function.
    # Register r14 will be stored in the slot reserved for compiler use.
    stg     %r14, 8(%r15)
    std     %f0, 128(%r15)
    std     %f2, 136(%r15)
    std     %f4, 144(%r15)
    std     %f6, 152(%r15)
    aghi    %r15, -STACKSZ

    lgrl    %r1, _ZN6__xray19XRayPatchedFunctionE@GOT
    ltg     %r1, 0(%r1)
    je      .Lrestore0

    # Set r3 to XRayEntryType::ENTRY = 0.
    # The FuncId is still stored in r2.
    lghi    %r3, 0
    basr    %r14, %r1

.Lrestore0:
    ld      %f6, STACKSZ+152(%r15)
    ld      %f4, STACKSZ+144(%r15)
    ld      %f2, STACKSZ+136(%r15)
    ld      %f0, STACKSZ+128(%r15)
    lmg     %r1, %r15, STACKSZ+8(%r15)
    br      %r1
.Lfunc_end0:
    .size    __xray_FunctionEntry, .Lfunc_end0-__xray_FunctionEntry

//===----------------------------------------------------------------------===//

    .globl  __xray_FunctionEntryVec
    .p2align    4
    .type   __xray_FunctionEntryVec,@function
__xray_FunctionEntryVec:
    # The registers r2-15 of the instrumented function are already saved in the
    # stack frame. On entry, r2 contains the function id, and %r14 the address
    # of the first instruction of the instrumented function.
    # Register r14 will be stored in the slot reserved for compiler use.
    stg     %r14, 8(%r15)
    std     %f0, 128(%r15)
    std     %f2, 136(%r15)
    std     %f4, 144(%r15)
    std     %f6, 152(%r15)
    aghi    %r15, -STACKSZ_VEC
    vstm    %v24, %v31, 160(%r15)

    lgrl    %r1, _ZN6__xray19XRayPatchedFunctionE@GOT
    ltg     %r1, 0(%r1)
    je      .Lrestore1

    # Set r3 to XRayEntryType::ENTRY = 0.
    # The FuncId is still stored in r2.
    lghi    %r3, 0
    basr    %r14, %r1

.Lrestore1:
    vlm     %v24, %v31, 160(%r15)
    ld      %f6, STACKSZ_VEC+152(%r15)
    ld      %f4, STACKSZ_VEC+144(%r15)
    ld      %f2, STACKSZ_VEC+136(%r15)
    ld      %f0, STACKSZ_VEC+128(%r15)
    lmg     %r1, %r15, STACKSZ_VEC+8(%r15)
    br      %r1
.Lfunc_end1:
    .size    __xray_FunctionEntryVec, .Lfunc_end1-__xray_FunctionEntryVec

//===----------------------------------------------------------------------===//

    .globl  __xray_FunctionExit
    .p2align    4
    .type   __xray_FunctionExit,@function
__xray_FunctionExit:
    # The registers r2-15 of the instrumented function are already saved in the
    # stack frame. On entry, the register r2 contains the function id.
    # At the end, the function jumps to the address saved in the slot for r14,
    # which contains the return address into the caller of the instrumented
    # function.
    std     %f0, 128(%r15)
    std     %f2, 136(%r15)
    std     %f4, 144(%r15)
    std     %f6, 152(%r15)
    aghi    %r15, -STACKSZ

    lgrl    %r1, _ZN6__xray19XRayPatchedFunctionE@GOT
    ltg     %r1, 0(%r1)
    je      .Lrestore2

    # Set r3 to XRayEntryType::EXIT = 1.
    # The FuncId is still stored in r2.
    lghi    %r3, 1
    basr    %r14, %r1

.Lrestore2:
    ld      %f6, STACKSZ+152(%r15)
    ld      %f4, STACKSZ+144(%r15)
    ld      %f2, STACKSZ+136(%r15)
    ld      %f0, STACKSZ+128(%r15)
    lmg     %r2, %r15, STACKSZ+16(%r15)
    br      %r14
.Lfunc_end2:
    .size    __xray_FunctionExit, .Lfunc_end2-__xray_FunctionExit

//===----------------------------------------------------------------------===//

    .globl  __xray_FunctionExitVec
    .p2align    4
    .type   __xray_FunctionExitVec,@function
__xray_FunctionExitVec:
    # The registers r2-15 of the instrumented function are already saved in the
    # stack frame. On entry, the register r2 contains the function id.
    # At the end, the function jumps to the address saved in the slot for r14,
    # which contains the return address into the caller of the instrumented
    # function.
    std     %f0, 128(%r15)
    std     %f2, 136(%r15)
    std     %f4, 144(%r15)
    std     %f6, 152(%r15)
    aghi    %r15, -STACKSZ_VEC
    vstm    %v24, %v31, 160(%r15)

    lgrl    %r1, _ZN6__xray19XRayPatchedFunctionE@GOT
    ltg     %r1, 0(%r1)
    je      .Lrestore3

    # Set r3 to XRayEntryType::EXIT = 1.
    # The FuncId is still stored in r2.
    lghi    %r3, 1
    basr    %r14, %r1

.Lrestore3:
    vlm     %v24, %v31, 160(%r15)
    ld      %f6, STACKSZ_VEC+152(%r15)
    ld      %f4, STACKSZ_VEC+144(%r15)
    ld      %f2, STACKSZ_VEC+136(%r15)
    ld      %f0, STACKSZ_VEC+128(%r15)
    lmg     %r2, %r15, STACKSZ_VEC+16(%r15)
    br      %r14
.Lfunc_end3:
    .size    __xray_FunctionExit, .Lfunc_end3-__xray_FunctionExit

//===----------------------------------------------------------------------===//

    .section    ".note.GNU-stack","",@progbits
