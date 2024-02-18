// Please note the following:
//   + we are checking that the first bytes of the PPA2 are 0x3 0x0
//     for C, and 0x3 0x1 for C++
//   + the label for the PPA2 seems to vary on different versions.
//     We try to cover all cases, and use substitution blocks to
//     help write the tests. The contents of the PPA2 itself should
//     not be different.
//   + the [[:space:]] combines the two .byte lines into one pattern.
//     This is necessary because if the lines were separated, the first
//     .byte (i.e., the one for the 3) would, it seems, also match
//     the .byte line below for the 34.

// REQUIRES: systemz-registered-target

// RUN: %clang_cc1 -triple s390x-ibm-zos -xc -S -o - %s | FileCheck %s --check-prefix CHECK-C
// CHECK-C:        [[PPA2:(.L)|(@@)PPA2]]:
// CHECK-C-NEXT:   .byte        3{{[[:space:]]*}}.byte 0
// CHECK-C-NEXT:   .byte        34{{$}}
// CHECK-C-NEXT:   .byte        {{4}}
// CHECK-C-NEXT:   .long        {{(CELQSTRT)}}-[[PPA2]]

// RUN: %clang_cc1 -triple s390x-ibm-zos -xc++ -S -o - %s | FileCheck %s --check-prefix CHECK-CXX
// CHECK-CXX:        [[PPA2:(.L)|(@@)PPA2]]:
// CHECK-CXX-NEXT:   .byte      3{{[[:space:]]*}}.byte 1
// CHECK-CXX-NEXT:   .byte      34{{$}}
// CHECK-CXX-NEXT:   .byte      {{4}}
// CHECK-CXX-NEXT:   .long      {{(CELQSTRT)}}-[[PPA2]]
