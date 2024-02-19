// Test the driver's control over the PIC behavior for PS5 compiler.
// These consist of tests of the relocation model flags and the
// pic level flags passed to CC1.
//
// CHECK-NO-PIC: "-mrelocation-model" "static"
// CHECK-NO-PIC-NOT: "-pic-level"
// CHECK-NO-PIC-NOT: "-pic-is-pie"
//
// CHECK-DYNAMIC-NO-PIC2: unsupported option '-mdynamic-no-pic'
// CHECK-DYNAMIC-NO-PIC2: "-mrelocation-model" "dynamic-no-pic"
//
// CHECK-PIC2: "-mrelocation-model" "pic"
// CHECK-PIC2: "-pic-level" "2"
//
// CHECK-PIE2: "-mrelocation-model" "pic"
// CHECK-PIE2: "-pic-is-pie"
//
// CHECK-NOPIC-IGNORED: using '-fPIC'
// CHECK-NOPIC-IGNORED: "-mrelocation-model" "pic"
// CHECK-NOPIC-IGNORED: "-pic-level" "2"
//
// CHECK-DIAG-PIC: option '-fno-PIC' was ignored by the PS5 toolchain, using '-fPIC'
// CHECK-DIAG-PIE: option '-fno-PIE' was ignored by the PS5 toolchain, using '-fPIC'
// CHECK-DIAG-pic: option '-fno-pic' was ignored by the PS5 toolchain, using '-fPIC'
// CHECK-DIAG-pie: option '-fno-pie' was ignored by the PS5 toolchain, using '-fPIC'
//
// CHECK-STATIC-ERR: unsupported option '-static' for target 'PS5'

// RUN: %clang -c %s -target x86_64-sie-ps5 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIC -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIC -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpie -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIE -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpie -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIE -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fPIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpic -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fpie -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Defaults change for PS5.
// RUN: %clang -c %s -target x86_64-sie-ps5 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
//
// Disregard any of the PIC-specific flags if we have a trump-card flag.
// RUN: %clang -c %s -target x86_64-sie-ps5 -mkernel -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: not %clang -c %s --target=x86_64-sie-ps5 -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC2
//
// -static not supported at all.
// RUN: not %clang -c %s --target=x86_64-sie-ps5 -static -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-STATIC-ERR
//
// -fno-PIC etc. is obeyed if -mcmodel=kernel is also present.
// RUN: %clang -c %s -target x86_64-sie-ps5 -mcmodel=kernel -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-sie-ps5 -mcmodel=kernel -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-sie-ps5 -mcmodel=kernel -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-sie-ps5 -mcmodel=kernel -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// Verify that we reflect the option the user specified, when we ignore it.
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-PIC
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-PIE
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-pic
// RUN: %clang -c %s -target x86_64-sie-ps5 -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-pie
