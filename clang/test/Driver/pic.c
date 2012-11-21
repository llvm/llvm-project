// Test the driver's control over the PIC behavior. These consist of tests of
// the relocation model flags and the pic level flags passed to CC1.
//
// CHECK-NO-PIC: "-mrelocation-model" "static"
// CHECK-NO-PIC-NOT: "-pic-level"
// CHECK-NO-PIC-NOT: "-pie-level"
//
// CHECK-DYNAMIC-NO-PIC1: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC1: "-pic-level" "1"
//
// CHECK-DYNAMIC-NO-PIC2: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC2: "-pic-level" "2"
//
// CHECK-PIC1-NOT: "-mrelocation-model"
// CHECK-PIC1: "-pic-level" "1"
//
// CHECK-PIC2-NOT: "-mrelocation-model"
// CHECK-PIC2: "-pic-level" "2"
//
// CHECK-PIE1-NOT: "-mrelocation-model"
// CHECK-PIE1: "-pie-level" "1"
//
// CHECK-PIE2-NOT: "-mrelocation-model"
// CHECK-PIE2: "-pie-level" "2"
//
// CHECK-PIE3: "{{.*}}ld{{(.exe)?}}"
// CHECK-PIE3: "-pie"
// CHECK-PIE3: "Scrt1.o" "crti.o" "crtbeginS.o"
// CHECK-PIE3: "crtendS.o" "crtn.o"
//
// RUN: %clang -c %s -target i386-unknown-unknown -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Cases where both pic and pie are specified
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Defaults change for Darwin.
// RUN: %clang -c %s -target i386-apple-darwin -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// Make sure -pie is passed to along to ld and that the right *crt* files
// are linked in.
// RUN: %clang %s -target i386-unknown-freebsd -fPIE -pie -### \
// RUN: --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE3
// RUN: %clang %s -target i386-linux-gnu -fPIE -pie -### \
// RUN: --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE3
//
// Disregard any of the PIC-specific flags if we have a trump-card flag.
// RUN: %clang -c %s -target i386-unknown-unknown -mkernel -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -static -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC1
// RUN: %clang -c %s -target i386-apple-darwin -mdynamic-no-pic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC2

// Checks for ARM
// RUN: %clang -c %s -target armv7-apple-ios -fapple-kext -miphoneos-version-min=6.0.0 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target armv7-apple-ios -mkernel -miphoneos-version-min=6.0.0 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target armv7-apple-ios -fapple-kext -miphoneos-version-min=5.0.0 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target armv7-apple-ios -fapple-kext -miphoneos-version-min=6.0.0 -static -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
