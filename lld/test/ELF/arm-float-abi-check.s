// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi hf.s -o hf.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi softfp.s -o softfp.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi compat.s -o compat.o
// RUN: not ld.lld hf.o softfp.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR -DF1=softfp.o -DF2=hf.o
// RUN: ld.lld hf.o compat.o -o /dev/null
// RUN: ld.lld softfp.o compat.o -o /dev/null

// ERR: error: [[F1]]: cannot link object files with different floating-point ABI from [[F2]]

//--- hf.s
.syntax unified
.global _start
.type _start, %function
_start:
    bx lr
.eabi_attribute 10, 2 // Tag_FP_arch = VFPv2
.eabi_attribute 28, 1 // Tag_ABI_VFP_args = AAPCS VFP (hard float)

//--- softfp.s
.syntax unified
.global f
.type f, %function
f:
    bx lr
.eabi_attribute 10, 2 // Tag_FP_arch = VFPv2
// Tag_ABI_VFP_args is omitted (implicitly BaseAAPCS, i.e., softfp since Tag_FP_arch is set)

//--- compat.s
.syntax unified
.global g
.type g, %function
g:
    bx lr
.eabi_attribute 28, 3 // Tag_ABI_VFP_args = CompatibleFPAAPCS (compatible with all)
