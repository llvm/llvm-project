// REQUIRES:arm

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-vfp-arg-base.s -o %t/base.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-vfp-arg-vfp.s -o %t/vfp.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-vfp-arg-toolchain.s -o %t/toolchain.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %t/main.s -o %t/main.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %t/vendor.s -o %t/vendor.o
// RUN: not ld.lld %t/main.o %t/base.o %t/vfp.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/base.o %t/vendor.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/base.o %t/toolchain.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/vfp.o %t/base.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/vfp.o %t/toolchain.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/toolchain.o %t/base.o -o%t/a.out 2>&1 | FileCheck %s
// RUN: not ld.lld %t/main.o %t/toolchain.o %t/vfp.o -o%t/a.out 2>&1 | FileCheck %s

// CHECK: incompatible Tag_ABI_VFP_args

//--- main.s

	.arch armv7-a
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 2
	.eabi_attribute 30, 6
	.eabi_attribute 34, 1
	.eabi_attribute 18, 4
        .eabi_attribute 28, 3 // Tag_ABI_VFP_args = 3 (Compatible with all)

        .syntax unified
        .globl _start
        .type _start, %function
_start: bx lr

//--- vendor.s

        .syntax unified

        // Manually construct a custom .ARM.attributes section
        .section .ARM.attributes,"",%0x70000003 // SHT_ARM_ATTRIBUTES

        // Initial byte giving the section format version
        .byte 'A'

        // Subsection with a name that won't be recognised as a known vendor
vendor_subsect_start:
        .word vendor_subsect_end - vendor_subsect_start // subsection length
        .asciz "ShouldBeIgnored" // vendor name
        .dcb.b 64, 0xff // dummy vendor section contents
vendor_subsect_end:

        // Subsection that should be identical to the attributes defined by
        // Inputs/arm-vfp-arg-vfp.s
aeabi_subsect_start:
        .word aeabi_subsect_end - aeabi_subsect_start
        .asciz "aeabi" // vendor name indicating the standard subsection
file_subsubsect_start:
        .byte 1 // introduce sub-subsection of attributes for the whole file
        .word file_subsubsect_end - file_subsubsect_start // sub-subsection len
        .byte 5 // CPU_name
        .asciz "7-A"
        .byte 6, 10 // CPU_arch = ARM v7
        .byte 7, 'A' // CPU_arch_profile = Application
        .byte 8, 1 // ARM_ISA_use = Permitted
        .byte 9, 2 // THUMB_ISA_use = Thumb-2
        .byte 18, 4 // ABI_PCS_wchar_t = 4-byte
        .byte 20, 1 // ABI_FP_denormal = IEEE-754
        .byte 21, 1 // ABI_FP_exceptions = IEEE-754
        .byte 23, 3 // ABI_FP_number_model = IEEE-754
        .byte 24, 1 // ABI_align_needed = 8-byte alignment
        .byte 25, 1 // ABI_align_preserved = 8-byte data alignment
        .byte 26, 2 // ABI_enum_size = Int32
        .byte 28, 1 // ABI_VFP_args = AAPCS VFP
        .byte 30, 6 // ABI_optimization_goals = Best Debugging
        .byte 34, 1 // CPU_unaligned_access = v6-style
file_subsubsect_end:
aeabi_subsect_end:

        .text
        .global f1
        .type f1, %function
f1:     bx lr
