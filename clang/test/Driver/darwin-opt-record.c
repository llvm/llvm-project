// REQUIRES: system-darwin

// RUN: %clang -target x86_64-apple-darwin10 -### -c -o FOO -fsave-optimization-record -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH
// RUN: %clang -target x86_64-apple-darwin10 -### -c -o FOO -foptimization-record-file=tmp -arch x86_64 -arch x86_64h %s 2>&1 | FileCheck %s --check-prefix=CHECK-MULTIPLE-ARCH-ERROR
// RUN: %clang -target x86_64-apple-darwin10 -### -o FOO -fsave-optimization-record %s 2>&1 | FileCheck %s --check-prefix=CHECK-DSYMUTIL-NO-G
// RUN: %clang -target x86_64-apple-darwin10 -### -o FOO -g0 -fsave-optimization-record %s 2>&1 | FileCheck %s --check-prefix=CHECK-DSYMUTIL-G0
// RUN: %clang -target x86_64-apple-darwin10 -### -o FOO -fsave-optimization-record=bitstream %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-G-PATH-BITSTREAM

// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64.opt.yaml"
// CHECK-MULTIPLE-ARCH: "-cc1"
// CHECK-MULTIPLE-ARCH: "-opt-record-file" "FOO-x86_64h.opt.yaml"

// CHECK-MULTIPLE-ARCH-ERROR: cannot use '-foptimization-record-file' output with multiple -arch options

// CHECK-DSYMUTIL-NO-G: "-cc1"
// CHECK-DSYMUTIL-NO-G: ld
// CHECK-DSYMUTIL-NO-G: dsymutil

// Even in the presence of -g0, -fsave-optimization-record implies
// -gline-tables-only and would need -fno-save-optimization-record to
// completely disable it.
// CHECK-DSYMUTIL-G0: "-cc1"
// CHECK-DSYMUTIL-G0: ld
// CHECK-DSYMUTIL-G0: dsymutil

// CHECK-NO-G-PATH-BITSTREAM: "-cc1"
// CHECK-NO-G-PATH-BITSTREAM: "-opt-record-file" "[[OUTPATH:.*]].opt.bitstream"
// CHECK-NO-G-PATH-BITSTREAM: "-o" "[[OUTPATH:.*]].o"
// CHECK-NO-G-PATH-BITSTREAM: ld
// CHECK-NO-G-PATH-BITSTREAM: dsymutil
