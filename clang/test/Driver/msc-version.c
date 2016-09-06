//
// Verify -fms-compatibility-version parsing
//

// RUN: %clang -target i686-windows -fms-compatibility -fms-compatibility-version=14 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR

// CHECK-MSC-VERSION-MAJOR: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR: _MSC_FULL_VER 140000000
// CHECK-MSC-VERSION-MAJOR: _MSC_VER 1400

// RUN: %clang -target i686-windows -fms-compatibility -fms-compatibility-version=15.00 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR

// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_FULL_VER 150000000
// CHECK-MSC-VERSION-MAJOR-MINOR: _MSC_VER 1500

// RUN: %clang -target i686-windows -fms-compatibility -fms-compatibility-version=15.00.20706 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR-BUILD

// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_FULL_VER 150020706
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD: _MSC_VER 1500

// RUN: %clang -target i686-windows -fms-compatibility -fms-compatibility-version=15.00.20706.01 -dM -E - </dev/null -o - | FileCheck %s -check-prefix CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH

// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_BUILD 1
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_FULL_VER 150020706
// CHECK-MSC-VERSION-MAJOR-MINOR-BUILD-PATCH: _MSC_VER 1500


//
// Verify -fmsc-version and -fms-compatibility-version diagnostic
//

// RUN: not %clang -target i686-windows -fms-compatibility -fmsc-version=1700 -fms-compatibility-version=17.00.50727.1 -E - </dev/null 2>&1 | FileCheck %s -check-prefix CHECK-BASIC-EXTENDED-DIAGNOSTIC

// CHECK-BASIC-EXTENDED-DIAGNOSTIC: invalid argument '-fmsc-version={{.*}}' not allowed with '-fms-compatibility-version={{.*}}'


//
// Verify -fmsc-version to -fms-compatibility-version conversion
//

// RUN: %clang -### -target i686-windows -fms-compatibility -fmsc-version=17 -E - </dev/null -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MSC-17

// CHECK-MSC-17-NOT: "-fmsc-version=1700"
// CHECK-MSC-17: "-fms-compatibility-version=17"

// RUN: %clang -### -target i686-windows -fms-compatibility -fmsc-version=1600 -E - </dev/null -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MSC-16

// CHECK-MSC-16-NOT: "-fmsc-version=1600"
// CHECK-MSC-16: "-fms-compatibility-version=16.0"

// RUN: %clang -### -target i686-windows -fms-compatibility -fmsc-version=150020706 -E - </dev/null -o /dev/null 2>&1 | FileCheck %s -check-prefix CHECK-MSC-15

// CHECK-MSC-15-NOT: "-fmsc-version=150020706"
// CHECK-MSC-15: "-fms-compatibility-version=15.0.20706"

