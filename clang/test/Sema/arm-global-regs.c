// RUN: %clang_cc1 -ffreestanding -fsyntax-only -target-feature +reserve-r9  -verify -triple arm-arm-none-eabi %s

// Check a small subset of valid and invalid global register variable declarations.
// Also check that for global register variables without -ffixed-reg options it throws an error.

register unsigned arm_r3 __asm("r3"); //expected-error {{register 'r3' unsuitable for global register variables on this target}}

register unsigned arm_r12 __asm("r12"); //expected-error {{register 'r12' unsuitable for global register variables on this target}}

register unsigned arm_r5 __asm("r5"); //expected-error {{register 'r5' unsuitable for global register variables on this target}}

register unsigned arm_r9 __asm("r9");

register unsigned arm_r6 __asm("r6"); //expected-error {{-ffixed-r6 is required for global named register variable declaration}}

register unsigned arm_r7 __asm("r7"); //expected-error {{-ffixed-r7 is required for global named register variable declaration}}

register unsigned *parm_r7 __asm("r7"); //expected-error {{-ffixed-r7 is required for global named register variable declaration}}

register unsigned arm_sp __asm("sp");
