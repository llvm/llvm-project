// RUN: %clang_cc1 -emit-pch -D TWO=2 -D X=4 -o %t %s -triple spir-unknown-unknown
// RUN: %clang_cc1 -include-pch %t -D THREE=3 -D X=5 -O0 -U__OPTIMIZE__ -fno-validate-pch %s -triple spir-unknown-unknown 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -include-pch %t -D THREE=3 -D X=5 -D VALIDATE -O0 -fsyntax-only %s -triple spir-unknown-unknown 2>&1 | FileCheck --check-prefix=CHECK-VAL %s

#ifndef HEADER
#define HEADER
// Header.

#define ONE 1

#else
// Using the header.

// CHECK: warning: 'X' macro redefined
// CHECK: #define X 5
// CHECK: note: previous definition is here
// CHECK: #define X 4

// CHECK-VAL: error: __OPTIMIZE__ predefined macro was enabled in AST file '{{.*}}' but is currently disabled
// CHECK-VAL: error: definition of macro 'X' differs between the AST file '{{.*}}' ('4') and the command line ('5')

void test(void) {
  int a = ONE;
  int b = TWO;
  int c = THREE;

#ifndef VALIDATE
#if X != 5
#error Definition of X is not overridden!
#endif

#ifdef __OPTIMIZE__
#error Optimization is not off!
#endif
#endif

}

#endif
