# REQUIRES: x86

# RUN: llvm-mc -n -filetype=obj -triple=x86_64 %s -o %t.o

.globl _start
.section ._start,"ax",@progbits
_start:
jmp test_simple
.size _start, .-_start

# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_simple | FileCheck %s --check-prefix=SIMPLE
# SIMPLE:      live symbol: test_simple
# SIMPLE-NEXT: >>> kept alive by _start
.globl test_simple
.section .test_simple,"ax",@progbits
test_simple:
jmp test_simple
jmp test_from_unsized

## This is alive merely by virtue of being a member of test_simple.
# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_incidental | FileCheck %s --check-prefix=INCIDENTAL
# INCIDENTAL:          live symbol: test_incidental
# INCIDENTAL-NEXT: >>> kept alive by {{.*}}.o:(.test_simple)
# INCIDENTAL-NEXT: >>> kept alive by test_simple
# INCIDENTAL-NEXT: >>> kept alive by _start
.globl test_incidental
test_incidental:
jmp test_incidental

# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_from_unsized | FileCheck %s --check-prefix=FROM-UNSIZED
# FROM-UNSIZED:          live symbol: test_from_unsized
# FROM-UNSIZED-NEXT: >>> kept alive by /usr/local/google/home/dthorn/llvm-project/build/tools/lld/test/ELF/Output/why-live.s.tmp.o:(.test_simple)
# FROM-UNSIZED-NEXT: >>> kept alive by test_simple
# FROM-UNSIZED-NEXT: >>> kept alive by _start
.globl test_from_unsized
.section .test_from_unsized,"ax",@progbits
test_from_unsized:
jmp test_from_unsized

# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_dead | count 0
.globl test_dead
.section .test_dead,"ax",@progbits
test_dead:
jmp test_dead

## Undefined symbols are not considered live.
# RUN: ld.lld %t.o -o /dev/null --gc-sections --why-live=test_undef -u test_undef | count 0

