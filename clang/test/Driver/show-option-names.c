// REQUIRES: x86-registered-target

// RUN: %clang -target x86_64-apple-darwin -c -isysroot /FOO %s 2>&1 | FileCheck --check-prefix=CHECK-SHOW-OPTION-NAMES %s
// CHECK-SHOW-OPTION-NAMES: warning: no such sysroot directory: '{{([A-Za-z]:.*)?}}/FOO' [-Wmissing-sysroot]

// RUN: %clang -target x86_64-apple-darwin -c -fno-diagnostics-show-option -isysroot /FOO %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SHOW-OPTION-NAMES %s
// CHECK-NO-SHOW-OPTION-NAMES: warning: no such sysroot directory: '{{([A-Za-z]:.*)?}}/FOO'{{$}}
