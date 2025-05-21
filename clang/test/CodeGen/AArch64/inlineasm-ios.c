// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-apple-ios -S -o - %s | FileCheck %s

// CHECK:       _restartable_function:
// CHECK-NEXT:  	ldr	x11, [x0]
// CHECK-NEXT:  	add	x11, x11, #1
// CHECK-NEXT:  	str	x11, [x0]
// CHECK-NEXT:  Ltmp0:
// CHECK-NEXT:  	b	Ltmp0
// CHECK-NEXT:  LExit_restartable_function:
// CHECK-NEXT:  	ret
asm(".align 4\n"
    "    .text\n"
    "    .private_extern _restartable_function\n"
    "_restartable_function:\n"
    "    ldr    x11, [x0]\n"
    "    add    x11, x11, #1\n"
    "    str    x11, [x0]\n"
    "1:\n"
    "    b 1b\n"
    "LExit_restartable_function:\n"
    "    ret\n"
);
