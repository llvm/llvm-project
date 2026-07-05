# REQUIRES: arm

# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=armv7-unknown-linux %t/small.s -o %t.small.o
# RUN: llvm-mc -filetype=obj -triple=armv7-unknown-linux %t/large.s -o %t.large.o
# RUN: llvm-objcopy --set-section-flags .bar=alloc,readonly %t.large.o %t.large.RO.o

# RUN: echo ordered > %t_order.txt

# RUN: ld.lld --symbol-ordering-file %t_order.txt %t.small.o -o %t2.small.out
# RUN: ld.lld --symbol-ordering-file %t_order.txt %t.large.o -o %t2.large.out
# RUN: ld.lld --symbol-ordering-file %t_order.txt %t.large.RO.o -o %t2.large.RO.out
# RUN: llvm-nm -n %t2.small.out | FileCheck --check-prefix=SMALL %s
# RUN: llvm-nm -n %t2.large.out | FileCheck --check-prefix=LARGE %s
# RUN: llvm-nm -n %t2.large.RO.out | FileCheck --check-prefix=SMALL %s
# RUN: rm -f %t.*.o %t2.*.out

# SMALL: ordered
# SMALL-NEXT: unordered1
# SMALL-NEXT: unordered2
# SMALL-NEXT: unordered3
# SMALL-NEXT: unordered4

# LARGE: unordered1
# LARGE-NEXT: unordered2
# LARGE-NEXT: ordered
# LARGE-NEXT: unordered3
# LARGE-NEXT: unordered4

#--- small.s
.section .foo,"ax",%progbits,unique,1
unordered1:
.zero 1

.section .foo,"ax",%progbits,unique,2
unordered2:
.zero 1

.section .foo,"ax",%progbits,unique,3
unordered3:
.zero 2

.section .foo,"ax",%progbits,unique,4
unordered4:
.zero 4

.section .foo,"ax",%progbits,unique,5
ordered:
.zero 1

#--- large.s
.section .bar,"ax",%progbits,unique,1
unordered1:
.zero 0xC00000

.section .bar,"ax",%progbits,unique,2
unordered2:
.zero 0xC00000

.section .bar,"ax",%progbits,unique,3
unordered3:
.zero 0xC00000

.section .bar,"ax",%progbits,unique,4
unordered4:
.zero 0xC00000

.section .bar,"ax",%progbits,unique,5
ordered:
.zero 8
