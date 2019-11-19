// REQUIRES: system-darwin

// RUN: %clang -### -S -o FOO -fsave-optimization-record -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH
//
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64.opt.yaml"
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64h.opt.yaml"
