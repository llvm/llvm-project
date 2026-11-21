// RUN: %clang_cc1 -dump-tokens %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -dump-tokens -x assembler-with-cpp %s 2>&1 | FileCheck %s --check-prefix=CHECK-ASM
// PR3808

// CHECK: identifier '$A'
// CHECK-ASM: identifier 'A'
$A

#define CAT(A,B) CAT_(A,B)
#define CAT_(A,B) A ## B
int CAT(X,CAT(1,$));
