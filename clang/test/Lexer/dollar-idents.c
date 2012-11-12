// RUN: %clang_cc1 -dump-tokens %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -dump-tokens -x assembler-with-cpp %s 2>&1 | FileCheck %s --check-prefix=ASM
// PR3808

// CHECK: identifier '$A'
// CHECK-ASM: identifier 'A'
$A
