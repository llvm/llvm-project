// REQUIRES: arm-registered-target
// RUN: not %clang_cc1 -triple armv7a-none-eabi -mrelocation-model pic -emit-obj %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR-XO-PIC %s
// ERR-XO-PIC: error: '-mlong-calls' with '-mexecute-only' is not supported for position-independent code

void __attribute__((target("+long-calls,+execute-only")))
call_xo_pic(void) {}

// RUN: not %clang_cc1 -triple armv7a-none-eabi -mrelocation-model ropi -emit-obj %s -o /dev/null 2>&1 | FileCheck --check-prefix=ERR-ROPI %s
// ERR-ROPI: error: '-mlong-calls' is not supported with ROPI

void __attribute__((target("+long-calls")))
call_ropi(void) {}
