# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/input.s -o %t/input.o
# RUN: llvm-profdata merge %t/weighted.proftext %t/functions.proftext -o %t/weighted.profdata
# RUN: llvm-profdata merge %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext \
# RUN:   %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext %t/primary.proftext \
# RUN:   %t/competing.proftext %t/functions.proftext -o %t/replicated.profdata
# RUN: llvm-profdata merge %t/primary.proftext %t/competing.proftext %t/functions.proftext -o %t/unweighted.profdata

# A weight of 10 is equivalent to ten copies of the same trace in the ELF
# balanced-partitioning consumer.
# RUN: ld.lld -e _start -o %t/weighted.out %t/input.o --irpgo-profile=%t/weighted.profdata --bp-startup-sort=function
# RUN: ld.lld -e _start -o %t/replicated.out %t/input.o --irpgo-profile=%t/replicated.profdata --bp-startup-sort=function
# RUN: ld.lld -e _start -o %t/unweighted.out %t/input.o --irpgo-profile=%t/unweighted.profdata --bp-startup-sort=function
# RUN: llvm-nm -jn %t/weighted.out > %t/weighted.order
# RUN: llvm-nm -jn %t/replicated.out > %t/replicated.order
# RUN: llvm-nm -jn %t/unweighted.out > %t/unweighted.order
# RUN: cmp %t/weighted.order %t/replicated.order
# RUN: not cmp %t/weighted.order %t/unweighted.order
# RUN: FileCheck %s --input-file=%t/weighted.order --check-prefix=FUNCTIONS

# FUNCTIONS-DAG: A
# FUNCTIONS-DAG: B
# FUNCTIONS-DAG: C
# FUNCTIONS-DAG: D
# FUNCTIONS-DAG: E
# FUNCTIONS-DAG: F

#--- input.s
.section .text._start,"ax",@progbits
.globl _start
_start:
  ret

.section .text.A,"ax",@progbits
.globl A
A:
  ret

.section .text.B,"ax",@progbits
.globl B
B:
  ret

.section .text.C,"ax",@progbits
.globl C
C:
  ret

.section .text.D,"ax",@progbits
.globl D
D:
  ret

.section .text.E,"ax",@progbits
.globl E
E:
  ret

.section .text.F,"ax",@progbits
.globl F
F:
  ret

#--- weighted.proftext
:ir
:temporal_prof_traces
# Num Traces
2
# Trace Stream Size
11
# Weight
10
A, B, C, D, E, F
# Weight
1
A, D, B, C, E, F

#--- primary.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size
1
# Weight
1
A, B, C, D, E, F

#--- competing.proftext
:ir
:temporal_prof_traces
# Num Traces
1
# Trace Stream Size
1
# Weight
1
A, D, B, C, E, F

#--- functions.proftext
:ir
A
# Func Hash
1
# Num Counters
1
# Counter Values
1
B
2
1
1
C
3
1
1
D
4
1
1
E
5
1
1
F
6
1
1
