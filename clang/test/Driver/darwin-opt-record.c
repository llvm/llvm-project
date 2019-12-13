// REQUIRES: system-darwin

// RUN: %clang -target x86_64-apple-darwin10 -### -c -o FOO -fsave-optimization-record -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH
// RUN: %clang -target x86_64-apple-darwin10 -### -c -o FOO -foptimization-record-file=tmp -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH-ERROR
//
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64.opt.yaml"
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64h.opt.yaml"
//
// CHECK-MULTIPLE-ARCH-ERROR: cannot use '-foptimization-record-file' output with multiple -arch options
