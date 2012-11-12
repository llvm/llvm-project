// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused-parameter %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fblocks -fsyntax-only -Wunused %s 2>&1 | FileCheck -check-prefix=CHECK-unused %s

int f0(int x,
       int y,
       int z __attribute__((unused))) {
  return x;
}

void f1() {
  (void)^(int x,
          int y,
          int z __attribute__((unused))) { return x; };
}

// Used when testing '-Wunused' to see that we only emit one diagnostic, and no
// warnings for the above cases.
static void achor() {};

// CHECK: 5:12: warning: unused parameter 'y'
// CHECK: 12:15: warning: unused parameter 'y'
// CHECK-unused: 1 warning generated

// RUN: %clang_cc1 -fblocks -fsyntax-only -Weverything %s 2>&1 | FileCheck -check-prefix=CHECK-everything %s
// RUN: %clang_cc1 -fblocks -fsyntax-only -Weverything -Werror %s 2>&1 | FileCheck -check-prefix=CHECK-everything-error %s
// RUN: %clang_cc1 -fblocks -fsyntax-only -Weverything -Wno-unused %s 2>&1 | FileCheck -check-prefix=CHECK-everything-no-unused %s
// CHECK-everything: 6 warnings generated
// CHECK-everything-error: 5 errors generated
// CHECK-everything-no-unused: 5 warnings generated

