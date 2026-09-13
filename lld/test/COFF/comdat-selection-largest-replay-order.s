# REQUIRES: x86

# A deferred symbol slot can contain both a child of one COMDAT and the leader
# of another. Leaders must be selected before any child is published, even when
# the child was encountered first.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-small.s -o %t.root-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/group-b.s -o %t.group-b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/group-a.s -o %t.group-a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-large.s -o %t.root-large.obj

# group-b registers its secondary definition of a before group-a registers a
# as a leader. Publishing that child first would incorrectly reject group-a.
# RUN: lld-link /force:multiple /dll /noentry /nodefaultlib /include:a \
# RUN:   %t.root-small.obj %t.group-b.obj %t.group-a.obj %t.root-large.obj \
# RUN:   /out:%t.child-first.dll
# RUN: llvm-objdump -s %t.child-first.dll | FileCheck %s

# The result must not depend on the order in which the leader and child
# providers entered the slot.
# RUN: lld-link /force:multiple /dll /noentry /nodefaultlib /include:a \
# RUN:   %t.root-small.obj %t.group-a.obj %t.group-b.obj %t.root-large.obj \
# RUN:   /out:%t.leader-first.dll
# RUN: llvm-objdump -s %t.leader-first.dll | FileCheck %s

# CHECK: Contents of section .data:
# CHECK: aaaaaaaa
# CHECK-NOT: bbbbbbbb
# CHECK-NOT: 11111111

#--- root-small.s
        .section .data$root, "dw", largest, root_leader
        .globl root_leader
root_leader:
        .byte 0

        # Keep a before b in the symbol table. Discarding this group therefore
        # schedules a before b for deferred replay.
        .globl a
a:
        .long 0x11111111

        .globl b
b:
        .byte 0

#--- group-b.s
        .section .data$b, "dw", largest, b
        .globl b
b:
        .space 8

        .globl a
a:
        .long 0xbbbbbbbb

#--- group-a.s
        .section .data$a, "dw", largest, a
        .globl a
a:
        .long 0xaaaaaaaa

#--- root-large.s
        .section .data$root, "dw", largest, root_leader
        .globl root_leader
root_leader:
        .space 32
