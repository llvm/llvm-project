# REQUIRES: x86
## Check that group members are retained or discarded as a unit, and
## non-SHF_ALLOC sections in a group are subject to garbage collection.
## This is compatible with GNU ld.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t.dead
# RUN: llvm-readobj -S %t.dead | FileCheck %s --check-prefix=CHECK-DEAD

## .mynote.bar is retained because it is not in a group.
# CHECK-DEAD-NOT: Name: .myanote.foo
# CHECK-DEAD-NOT: Name: .mytext.foo
# CHECK-DEAD-NOT: Name: .mybss.foo
# CHECK-DEAD-NOT: Name: .mynote.foo
# CHECK-DEAD:     Name: .mynote.bar

# RUN: ld.lld --gc-sections %t.o -o %t -e anote_foo
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE
# RUN: ld.lld --gc-sections %t.o -o %t -e foo
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE
# RUN: ld.lld --gc-sections %t.o -o %t -e bss_foo
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE

## note_foo as the entry point does not make much sense because it is defined
## in a non-SHF_ALLOC section. This is just to demonstrate the behavior.
# RUN: ld.lld --gc-sections %t.o -o %t -e note_foo
# RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=CHECK-LIVE

# CHECK-LIVE: Name: .myanote.foo
# CHECK-LIVE: Name: .mytext.foo
# CHECK-LIVE: Name: .mybss.foo
# CHECK-LIVE: Name: .mynote.foo
# CHECK-LIVE: Name: .mynote.bar

.globl anote_foo, foo, bss_foo, note_foo

.section .myanote.foo,"aG",@note,foo,comdat
anote_foo:
.byte 0

.section .mytext.foo,"axG",@progbits,foo,comdat
foo:
.byte 0

.section .mybss.foo,"awG",@nobits,foo,comdat
bss_foo:
.byte 0

.section .mynote.foo,"G",@note,foo,comdat
note_foo:
.byte 0

.section .mynote.bar,"",@note
.byte 0
