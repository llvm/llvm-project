// RUN: (%clang -c -target i386-apple-darwin10 -isysroot /FOO %s 2>&1 || true) | FileCheck --check-prefix=CHECK-SHOW-OPTION-NAMES %s
// CHECK-SHOW-OPTION-NAMES: warning: no such sysroot directory: '{{([A-Za-z]:.*)?}}/FOO' [-Wmissing-sysroot]

// RUN: (%clang -c -target i386-apple-darwin10 -fno-diagnostics-show-option -isysroot /FOO %s 2>&1 || true) | FileCheck --check-prefix=CHECK-NO-SHOW-OPTION-NAMES %s
// CHECK-NO-SHOW-OPTION-NAMES: warning: no such sysroot directory: '{{([A-Za-z]:.*)?}}/FOO'{{$}}
