// RUN: clang-format %s --Wno-error=unknown --style="{UnknownKey: true}" 2>&1 | FileCheck %s -check-prefix=CHECK
// RUN: not clang-format %s --style="{UnknownKey: true}" 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL

// CHECK: <command-line>:1:2: warning: unknown key 'UnknownKey'
// CHECK-NEXT: {UnknownKey: true}
// CHECK-NEXT: ^~~~~~~~~~
// CHECK-FAIL: <command-line>:1:2: error: unknown key 'UnknownKey'
// CHECK-FAIL-NEXT: {UnknownKey: true}
// CHECK-FAIL-NEXT: ^~~~~~~~~~

int i ;
