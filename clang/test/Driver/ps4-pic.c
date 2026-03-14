// Test the driver's control over the PIC behavior for PS4 compiler.
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
// CHECK-DIAG-PIC: option '-fno-PIC' was ignored by the PS4 toolchain, using '-fPIC'
// CHECK-DIAG-PIE: option '-fno-PIE' was ignored by the PS4 toolchain, using '-fPIC'
// CHECK-DIAG-pic: option '-fno-pic' was ignored by the PS4 toolchain, using '-fPIC'
// CHECK-DIAG-pie: option '-fno-pie' was ignored by the PS4 toolchain, using '-fPIC'

// RUN: %clang -c %s -target x86_64-scei-ps4 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIC -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIC -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpie -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIE -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpie -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIE -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fPIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpic -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fpie -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Defaults change for PS4.
// RUN: %clang -c %s -target x86_64-scei-ps4 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIC-IGNORED
//
// Disregard any of the PIC-specific flags if we have a trump-card flag.
// RUN: %clang -c %s -target x86_64-scei-ps4 -mkernel -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: not %clang -c %s --target=x86_64-scei-ps4 -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC2
//
// The -static argument *doesn't* override PIC: -static only affects
// linking, and -fPIC only affects code generation.
// RUN: %clang -c %s -target x86_64-scei-ps4 -static -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
//
// -fno-PIC etc. is obeyed if -mcmodel=kernel is also present.
// RUN: %clang -c %s -target x86_64-scei-ps4 -mcmodel=kernel -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-scei-ps4 -mcmodel=kernel -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-scei-ps4 -mcmodel=kernel -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target x86_64-scei-ps4 -mcmodel=kernel -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// Verify that we reflect the option the user specified, when we ignore it.
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-PIC
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-PIE
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-pic
// RUN: %clang -c %s -target x86_64-scei-ps4 -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DIAG-pie
