// Please note the following:
//   + we are checking that the first bytes of the PPA2 are 0x3 0x0
//     for C, and 0x3 0x1 for C++
//   + the label for the PPA2 seems to vary on different versions.
//     We try to cover all cases, and use substitution blocks to
//     help write the tests. The contents of the PPA2 itself should
//     not be different.

// REQUIRES: systemz-registered-target

// RUN: %clang_cc1 -triple s390x-ibm-zos -xc -S -o - %s | FileCheck %s --check-prefix CHECK-C
// CHECK-C:        [[PPA2:(.L)|(L#)PPA2]] DS 0H
// CHECK-C-NEXT:   DC  XL1'03'
// CHECK-C-NEXT:   DC  XL1'00'
// CHECK-C-NEXT:   DC  XL1'22'
// CHECK-C-NEXT:   DC  XL1'04'
// CHECK-C-NEXT:   DC AD(CELQSTRT-[[PPA2]])

// RUN: %clang_cc1 -triple s390x-ibm-zos -xc++ -S -o - %s | FileCheck %s --check-prefix CHECK-CXX
// CHECK-CXX:        [[PPA2:(.L)|(L#)PPA2]] DS 0H
// CHECK-CXX-NEXT:   DC  XL1'03'
// CHECK-CXX-NEXT:   DC  XL1'01'
// CHECK-CXX-NEXT:   DC  XL1'22'
// CHECK-CXX-NEXT:   DC  XL1'04'
// CHECK-CXX-NEXT:   DC AD(CELQSTRT-[[PPA2]])
