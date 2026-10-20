// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s

// RUN: cp %s %t.i
// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic %t.i 2>&1 | FileCheck %s --check-prefix=NO-WARNING --allow-empty
// RUN: %clang_cc1 -std=c99 -fsyntax-only -pedantic -x cpp-output %s 2>&1 | FileCheck %s --check-prefix=NO-WARNING --allow-empty

# 0 "zero"
// CHECK: line-directive-suppressed.c:[[@LINE-1]]:5: warning: {{.*}} [-Wgnu-line-marker]

# 1 "one" 1
// CHECK: zero:2:5: warning: {{.*}} [-Wgnu-line-marker]

# 2 "two" 1 3 4
// CHECK: one:3:5: warning: {{.*}} [-Wgnu-line-marker]

// NO-WARNING-NOT: warning:
