# REQUIRES: x86

# RUN: llvm-mc -n -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_* | FileCheck %s

# CHECK:      live symbol: test_a
# CHECK-NEXT: >>> kept alive by {{.*}}.o:(._start)
# CHECK-NEXT: >>> kept alive by _start

# CHECK:      live symbol: test_b
# CHECK-NEXT: >>> kept alive by {{.*}}.o:(.test_a)
# CHECK-NEXT: >>> kept alive by test_a
# CHECK-NEXT: >>> kept alive by {{.*}}.o:(._start)
# CHECK-NEXT: >>> kept alive by _start

.globl _start
.section ._start,"ax",@progbits
_start:
# DO NOT SUBMIT: If this reads, "jmp a", then LLD hangs.
jmp test_a

.globl test_a
.section .test_a,"ax",@progbits
test_a:
jmp test_a

# This is alive merely by virtue of being a member of test_a.
.globl test_b
test_b:
jmp test_b

