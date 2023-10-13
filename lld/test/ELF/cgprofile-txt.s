# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld -e A %t -o %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s --check-prefix=NOSORT

# RUN: echo "A B 10" > %t.call_graph
# RUN: echo "A B 10" >> %t.call_graph
# RUN: echo "Aa B 80" >> %t.call_graph
# RUN: echo "A C 40" >> %t.call_graph
# RUN: echo "B C 30" >> %t.call_graph
# RUN: echo "C D 90" >> %t.call_graph
# RUN: echo "PP TS 100" >> %t.call_graph
# RUN: echo "_init2 _init 24567837" >> %t.call_graph
# RUN: echo "TS QC 9001" >> %t.call_graph
# RUN: echo "TooManyPreds0 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds1 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds2 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds3 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds4 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds5 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds6 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds7 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds8 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds9 TooManyPreds 10" >> %t.call_graph
# RUN: echo "TooManyPreds10 TooManyPreds 11" >> %t.call_graph
# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph --call-graph-profile-sort=hfsort -o %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s
## --call-graph-profile-sort=hfsort is the default.
# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph -o %t2b
# RUN: cmp %t2 %t2b

# RUN: ld.lld -e A %t --call-graph-ordering-file %t.call_graph --call-graph-profile-sort=cdsort -o %t2
# RUN: llvm-readobj --symbols %t2 | FileCheck %s --check-prefix=CDSORT

# RUN: not ld.lld -e A %t --call-graph-ordering-file %t.call_graph --call-graph-profile-sort=sort \
# RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN

# UNKNOWN: error: unknown --call-graph-profile-sort= value: sort

    .section    .text.D,"ax",@progbits
D:
    retq

    .section    .text.C,"ax",@progbits
    .globl  C
C:
    retq

    .section    .text.B,"ax",@progbits
    .globl  B
B:
    retq

    .section    .text.A,"ax",@progbits
    .globl  A
A:
Aa:
    retq

    .section    .ponies,"ax",@progbits,unique,1
    .globl TS
TS:
    retq

    .section    .ponies,"ax",@progbits,unique,2
    .globl PP
PP:
    retq

    .section    .other,"ax",@progbits,unique,1
    .globl QC
QC:
    retq

    .section    .other,"ax",@progbits,unique,2
    .globl GB
GB:
    retq

    .section    .init,"ax",@progbits,unique,1
    .globl _init
_init:
    retq

    .section    .init,"ax",@progbits,unique,2
    .globl _init2
_init2:
    retq

    .section    .text.TooManyPreds,"ax",@progbits
TooManyPreds:
    retq
	retq
	retq
	retq
	retq
	retq
	retq
	retq
	retq
	retq

    .section    .text.TooManyPreds0,"ax",@progbits
TooManyPreds0:
    retq

    .section    .text.TooManyPreds1,"ax",@progbits
TooManyPreds1:
    retq

    .section    .text.TooManyPreds2,"ax",@progbits
TooManyPreds2:
    retq

    .section    .text.TooManyPreds3,"ax",@progbits
TooManyPreds3:
    retq

    .section    .text.TooManyPreds4,"ax",@progbits
TooManyPreds4:
    retq

    .section    .text.TooManyPreds5,"ax",@progbits
TooManyPreds5:
    retq

    .section    .text.TooManyPreds6,"ax",@progbits
TooManyPreds6:
    retq

    .section    .text.TooManyPreds7,"ax",@progbits
TooManyPreds7:
    retq

    .section    .text.TooManyPreds8,"ax",@progbits
TooManyPreds8:
    retq

    .section    .text.TooManyPreds9,"ax",@progbits
TooManyPreds9:
    retq

    .section    .text.TooManyPreds10,"ax",@progbits
TooManyPreds10:
    retq

# CHECK:          Name: D
# CHECK-NEXT:     Value: 0x201123
# CHECK:          Name: TooManyPreds
# CHECK-NEXT:     Value: 0x201124
# CHECK:          Name: TooManyPreds10
# CHECK-NEXT:     Value: 0x201138
# CHECK:          Name: C
# CHECK-NEXT:     Value: 0x201122
# CHECK:          Name: B
# CHECK-NEXT:     Value: 0x201121
# CHECK:          Name: A
# CHECK-NEXT:     Value: 0x201120
# CHECK:          Name: TS
# CHECK-NEXT:     Value: 0x20113D
# CHECK:          Name: PP
# CHECK-NEXT:     Value: 0x20113C
# CHECK:          Name: QC
# CHECK-NEXT:     Value: 0x20113E
# CHECK:          Name: GB
# CHECK-NEXT:     Value: 0x20113F
# CHECK:          Name: _init
# CHECK-NEXT:     Value: 0x201140
# CHECK:          Name: _init2
# CHECK-NEXT:     Value: 0x201141

# CDSORT:          Name: D
# CDSORT-NEXT:     Value: 0x201123
# CDSORT:          Name: TooManyPreds
# CDSORT-NEXT:     Value: 0x20112F
# CDSORT:          Name: TooManyPreds10
# CDSORT-NEXT:     Value: 0x20112E
# CDSORT:          Name: C
# CDSORT-NEXT:     Value: 0x201122
# CDSORT:          Name: B
# CDSORT-NEXT:     Value: 0x201121
# CDSORT:          Name: A
# CDSORT-NEXT:     Value: 0x201120
# CDSORT:          Name: TS
# CDSORT-NEXT:     Value: 0x20113D
# CDSORT:          Name: PP
# CDSORT-NEXT:     Value: 0x20113C
# CDSORT:          Name: QC
# CDSORT-NEXT:     Value: 0x20113E
# CDSORT:          Name: GB
# CDSORT-NEXT:     Value: 0x20113F
# CDSORT:          Name: _init
# CDSORT-NEXT:     Value: 0x201140
# CDSORT:          Name: _init2
# CDSORT-NEXT:     Value: 0x201141

# NOSORT:          Name: D
# NOSORT-NEXT:     Value: 0x201120
# NOSORT:          Name: TooManyPreds
# NOSORT-NEXT:     Value: 0x201124
# NOSORT:          Name: TooManyPreds10
# NOSORT-NEXT:     Value: 0x201138
# NOSORT:          Name: C
# NOSORT-NEXT:     Value: 0x201121
# NOSORT:          Name: B
# NOSORT-NEXT:     Value: 0x201122
# NOSORT:          Name: A
# NOSORT-NEXT:     Value: 0x201123
# NOSORT:          Name: TS
# NOSORT-NEXT:     Value: 0x201139
# NOSORT:          Name: PP
# NOSORT-NEXT:     Value: 0x20113A
# NOSORT:          Name: QC
# NOSORT-NEXT:     Value: 0x20113B
# NOSORT:          Name: GB
# NOSORT-NEXT:     Value: 0x20113C
# NOSORT:          Name: _init
# NOSORT-NEXT:     Value: 0x20113D
# NOSORT:          Name: _init2
# NOSORT-NEXT:     Value: 0x20113E
