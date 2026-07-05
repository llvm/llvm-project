// REQUIRES: msp430-registered-target
// RUN: %clang_cc1 -mframe-pointer=all -triple msp430 -S %s -o - | FileCheck %s --check-prefix=FP_ENFORCED
// RUN: %clang_cc1 -triple msp430 -S %s -o - | FileCheck %s --check-prefix=FP_DEFAULT

// Check the frame pointer is not used on MSP430 by default, but can be forcibly turned on.

// FP_ENFORCED: push r4
// FP_ENFORCED: mov r4, r4
// FP_ENFORCED: pop r4
// FP_DEFAULT: .globl fp_elim_check
// FP_DEFAULT-NOT: push r4
// FP_DEFAULT: mov r4, r4
// FP_DEFAULT-NOT: pop r4

void fp_elim_check()
{
	asm volatile ("mov r4, r4");
}

