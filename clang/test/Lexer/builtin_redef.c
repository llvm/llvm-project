// RUN: %clang_cc1 %s -D__TIME__=1234 -U__DATE__ -D__STDC__=1 -U__STDC_HOSTED__ -E 2>&1 | FileCheck %s --check-prefix=CHECK-OUT
// RUN: %clang_cc1 %s -D__TIME__=1234 -U__DATE__ -D__STDC__=1 -U__STDC_HOSTED__ -E 2>&1 | FileCheck %s --check-prefix=CHECK-WARN
// RUN: not %clang_cc1 %s -D__TIME__=1234 -U__DATE__ -D__STDC__=1 -U__STDC_HOSTED__ -E 2>&1 -pedantic-errors | FileCheck %s --check-prefix=CHECK-ERR

// CHECK-WARN: <command line>:{{.*}} warning: redefining builtin macro
// CHECK-WARN-NEXT: #define __TIME__ 1234
// CHECK-WARN: <command line>:{{.*}} warning: undefining builtin macro
// CHECK-WARN-NEXT: #undef __DATE__
// CHECK-WARN: <command line>:{{.*}} warning: redefining builtin macro
// CHECK-WARN-NEXT: #define __STDC__ 1
// CHECK-WARN: <command line>:{{.*}} warning: undefining builtin macro
// CHECK-WARN-NEXT: #undef __STDC_HOSTED__

// CHECK-ERR: <command line>:{{.*}} error: redefining builtin macro
// CHECK-ERR-NEXT: #define __TIME__ 1234
// CHECK-ERR: <command line>:{{.*}} error: undefining builtin macro
// CHECK-ERR-NEXT: #undef __DATE__
// CHECK-ERR: <command line>:{{.*}} error: redefining builtin macro
// CHECK-ERR-NEXT: #define __STDC__ 1
// CHECK-ERR: <command line>:{{.*}} error: undefining builtin macro
// CHECK-ERR-NEXT: #undef __STDC_HOSTED__

int n = __TIME__;
__DATE__

#define __FILE__ "my file"
// CHECK-WARN: :[[@LINE-1]]:9: warning: redefining builtin macro
// CHECK-ERR: :[[@LINE-2]]:9: error: redefining builtin macro

// CHECK-OUT: int n = 1234;
// CHECK-OUT: __DATE__
