// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/resolution.s -o %t2
// RUN: ld.lld -discard-all %t %t2 -o %t3
// RUN: llvm-readelf --symbols %t3 | FileCheck %s

// This is an exhaustive test for checking which symbol is kept when two
// have the same name. Each symbol has a different size which is used
// to see which one was chosen.

// CHECK:      Symbol table '.symtab' contains 23 entries:
// CHECK-NEXT:  Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:     0 NOTYPE  LOCAL  DEFAULT   UND 
// CHECK-NEXT:     0 NOTYPE  GLOBAL DEFAULT     1 _start
// CHECK-NEXT:     0 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_RegularWeak
// CHECK-NEXT:    33 NOTYPE  GLOBAL DEFAULT     1 RegularWeak_with_RegularStrong
// CHECK-NEXT:     2 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_RegularWeak
// CHECK-NEXT:     3 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_UndefWeak
// CHECK-NEXT:     4 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_UndefStrong
// CHECK-NEXT:     5 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_UndefWeak
// CHECK-NEXT:     6 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_UndefStrong
// CHECK-NEXT:    40 OBJECT  GLOBAL DEFAULT     2 RegularWeak_with_CommonStrong
// CHECK-NEXT:    10 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_CommonStrong
// CHECK-NEXT:    43 NOTYPE  WEAK   DEFAULT     1 UndefWeak_with_RegularWeak
// CHECK-NEXT:    44 NOTYPE  GLOBAL DEFAULT     1 UndefWeak_with_RegularStrong
// CHECK-NEXT:    45 NOTYPE  WEAK   DEFAULT     1 UndefStrong_with_RegularWeak
// CHECK-NEXT:    46 NOTYPE  GLOBAL DEFAULT     1 UndefStrong_with_RegularStrong
// CHECK-NEXT:     0 NOTYPE  WEAK   DEFAULT   UND UndefWeak_with_UndefWeak
// CHECK-NEXT:    49 OBJECT  GLOBAL DEFAULT     2 UndefWeak_with_CommonStrong
// CHECK-NEXT:    51 OBJECT  GLOBAL DEFAULT     2 UndefStrong_with_CommonStrong
// CHECK-NEXT:    22 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_RegularWeak
// CHECK-NEXT:    55 NOTYPE  GLOBAL DEFAULT     1 CommonStrong_with_RegularStrong
// CHECK-NEXT:    26 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_UndefWeak
// CHECK-NEXT:    27 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_UndefStrong
// CHECK-NEXT:    63 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_CommonStrong

.globl _start
_start:
        nop

local:

.weak RegularWeak_with_RegularWeak
.size RegularWeak_with_RegularWeak, 0
RegularWeak_with_RegularWeak:

.weak RegularWeak_with_RegularStrong
.size RegularWeak_with_RegularStrong, 1
RegularWeak_with_RegularStrong:

.global RegularStrong_with_RegularWeak
.size RegularStrong_with_RegularWeak, 2
RegularStrong_with_RegularWeak:

.weak RegularWeak_with_UndefWeak
.size RegularWeak_with_UndefWeak, 3
RegularWeak_with_UndefWeak:

.weak RegularWeak_with_UndefStrong
.size RegularWeak_with_UndefStrong, 4
RegularWeak_with_UndefStrong:

.global RegularStrong_with_UndefWeak
.size RegularStrong_with_UndefWeak, 5
RegularStrong_with_UndefWeak:

.global RegularStrong_with_UndefStrong
.size RegularStrong_with_UndefStrong, 6
RegularStrong_with_UndefStrong:

.weak RegularWeak_with_CommonStrong
.size RegularWeak_with_CommonStrong, 8
RegularWeak_with_CommonStrong:

.global RegularStrong_with_CommonStrong
.size RegularStrong_with_CommonStrong, 10
RegularStrong_with_CommonStrong:

.weak UndefWeak_with_RegularWeak
.size UndefWeak_with_RegularWeak, 11
.quad UndefWeak_with_RegularWeak

.weak UndefWeak_with_RegularStrong
.size UndefWeak_with_RegularStrong, 12
.quad UndefWeak_with_RegularStrong

.size UndefStrong_with_RegularWeak, 13
.quad UndefStrong_with_RegularWeak

.size UndefStrong_with_RegularStrong, 14
.quad UndefStrong_with_RegularStrong

.weak UndefWeak_with_UndefWeak
.size UndefWeak_with_UndefWeak, 15
.quad UndefWeak_with_UndefWeak

.weak UndefWeak_with_CommonStrong
.size UndefWeak_with_CommonStrong, 17
.quad UndefWeak_with_CommonStrong

.size UndefStrong_with_CommonStrong, 19
.quad UndefStrong_with_CommonStrong

.comm CommonStrong_with_RegularWeak,22,4

.comm CommonStrong_with_RegularStrong,23,4

.comm CommonStrong_with_UndefWeak,26,4

.comm CommonStrong_with_UndefStrong,27,4

.comm CommonStrong_with_CommonStrong,31,4
