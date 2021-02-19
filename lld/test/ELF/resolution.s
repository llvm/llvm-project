// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/resolution.s -o %t2
// RUN: ld.lld -discard-all %t %t2 -o %t3
// RUN: llvm-readelf --symbols %t3 | FileCheck %s

// This is an exhaustive test for checking which symbol is kept when two
// have the same name. Each symbol has a different size which is used
// to see which one was chosen.

// CHECK:      Symbol table '.symtab' contains 34 entries:
// CHECK-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
// CHECK-NEXT:   1: 0000000000201158     0 NOTYPE  GLOBAL DEFAULT     1 _start
// CHECK-NEXT:   2: 0000000000201159     0 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_RegularWeak
// CHECK-NEXT:   3: 00000000002011a4    33 NOTYPE  GLOBAL DEFAULT     1 RegularWeak_with_RegularStrong
// CHECK-NEXT:   4: 0000000000201159     2 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_RegularWeak
// CHECK-NEXT:   5: 0000000000201159     3 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_UndefWeak
// CHECK-NEXT:   6: 0000000000201159     4 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_UndefStrong
// CHECK-NEXT:   7: 0000000000201159     5 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_UndefWeak
// CHECK-NEXT:   8: 0000000000201159     6 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_UndefStrong
// CHECK-NEXT:   9: 0000000000201159     7 NOTYPE  WEAK   DEFAULT     1 RegularWeak_with_CommonWeak
// CHECK-NEXT:  10: 00000000002021ec    40 OBJECT  GLOBAL DEFAULT     2 RegularWeak_with_CommonStrong
// CHECK-NEXT:  11: 0000000000201159     9 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_CommonWeak
// CHECK-NEXT:  12: 0000000000201159    10 NOTYPE  GLOBAL DEFAULT     1 RegularStrong_with_CommonStrong
// CHECK-NEXT:  13: 00000000002011c4    43 NOTYPE  WEAK   DEFAULT     1 UndefWeak_with_RegularWeak
// CHECK-NEXT:  14: 00000000002011c4    44 NOTYPE  GLOBAL DEFAULT     1 UndefWeak_with_RegularStrong
// CHECK-NEXT:  15: 00000000002011c4    45 NOTYPE  WEAK   DEFAULT     1 UndefStrong_with_RegularWeak
// CHECK-NEXT:  16: 00000000002011c4    46 NOTYPE  GLOBAL DEFAULT     1 UndefStrong_with_RegularStrong
// CHECK-NEXT:  17: 0000000000000000     0 NOTYPE  WEAK   DEFAULT   UND UndefWeak_with_UndefWeak
// CHECK-NEXT:  18: 0000000000202214    48 OBJECT  WEAK   DEFAULT     2 UndefWeak_with_CommonWeak
// CHECK-NEXT:  19: 0000000000202244    49 OBJECT  GLOBAL DEFAULT     2 UndefWeak_with_CommonStrong
// CHECK-NEXT:  20: 0000000000202278    50 OBJECT  WEAK   DEFAULT     2 UndefStrong_with_CommonWeak
// CHECK-NEXT:  21: 00000000002022ac    51 OBJECT  GLOBAL DEFAULT     2 UndefStrong_with_CommonStrong
// CHECK-NEXT:  22: 00000000002022e0    20 OBJECT  WEAK   DEFAULT     2 CommonWeak_with_RegularWeak
// CHECK-NEXT:  23: 00000000002011cc    53 NOTYPE  GLOBAL DEFAULT     1 CommonWeak_with_RegularStrong
// CHECK-NEXT:  24: 00000000002022f4    22 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_RegularWeak
// CHECK-NEXT:  25: 00000000002011cc    55 NOTYPE  GLOBAL DEFAULT     1 CommonStrong_with_RegularStrong
// CHECK-NEXT:  26: 000000000020230c    24 OBJECT  WEAK   DEFAULT     2 CommonWeak_with_UndefWeak
// CHECK-NEXT:  27: 0000000000202324    25 OBJECT  WEAK   DEFAULT     2 CommonWeak_with_UndefStrong
// CHECK-NEXT:  28: 0000000000202340    26 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_UndefWeak
// CHECK-NEXT:  29: 000000000020235c    27 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_UndefStrong
// CHECK-NEXT:  30: 0000000000202378    28 OBJECT  WEAK   DEFAULT     2 CommonWeak_with_CommonWeak
// CHECK-NEXT:  31: 0000000000202394    61 OBJECT  GLOBAL DEFAULT     2 CommonWeak_with_CommonStrong
// CHECK-NEXT:  32: 00000000002023d4    30 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_CommonWeak
// CHECK-NEXT:  33: 00000000002023f4    63 OBJECT  GLOBAL DEFAULT     2 CommonStrong_with_CommonStrong

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

.weak RegularWeak_with_CommonWeak
.size RegularWeak_with_CommonWeak, 7
RegularWeak_with_CommonWeak:

.weak RegularWeak_with_CommonStrong
.size RegularWeak_with_CommonStrong, 8
RegularWeak_with_CommonStrong:

.global RegularStrong_with_CommonWeak
.size RegularStrong_with_CommonWeak, 9
RegularStrong_with_CommonWeak:

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

.weak UndefWeak_with_CommonWeak
.size UndefWeak_with_CommonWeak, 16
.quad UndefWeak_with_CommonWeak

.weak UndefWeak_with_CommonStrong
.size UndefWeak_with_CommonStrong, 17
.quad UndefWeak_with_CommonStrong

.size UndefStrong_with_CommonWeak, 18
.quad UndefStrong_with_CommonWeak

.size UndefStrong_with_CommonStrong, 19
.quad UndefStrong_with_CommonStrong

.weak CommonWeak_with_RegularWeak
.comm CommonWeak_with_RegularWeak,20,4

.weak CommonWeak_with_RegularStrong
.comm CommonWeak_with_RegularStrong,21,4

.comm CommonStrong_with_RegularWeak,22,4

.comm CommonStrong_with_RegularStrong,23,4

.weak CommonWeak_with_UndefWeak
.comm CommonWeak_with_UndefWeak,24,4

.weak CommonWeak_with_UndefStrong
.comm CommonWeak_with_UndefStrong,25,4

.comm CommonStrong_with_UndefWeak,26,4

.comm CommonStrong_with_UndefStrong,27,4

.weak CommonWeak_with_CommonWeak
.comm CommonWeak_with_CommonWeak,28,4

.weak CommonWeak_with_CommonStrong
.comm CommonWeak_with_CommonStrong,29,4

.comm CommonStrong_with_CommonWeak,30,4

.comm CommonStrong_with_CommonStrong,31,4
