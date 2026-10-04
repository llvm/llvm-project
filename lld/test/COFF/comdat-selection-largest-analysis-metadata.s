# REQUIRES: x86

# Object-level addrsig and call-graph entries that name only a discarded
# secondary definition must have no observable ICF, GC, or layout effect.
# Exercise discard equivalence in both input orders.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /opt:icf \
# RUN:   /debug:symtab %t.large.obj %t.root.obj /out:%t.winner.exe
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /opt:icf \
# RUN:   /debug:symtab %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.small-first.exe
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /opt:icf \
# RUN:   /debug:symtab %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.large-first.exe
# RUN: llvm-nm --numeric-sort %t.winner.exe > %t.winner.nm
# RUN: llvm-nm --numeric-sort %t.small-first.exe > %t.small-first.nm
# RUN: llvm-nm --numeric-sort %t.large-first.exe > %t.large-first.nm
# RUN: cmp %t.winner.nm %t.small-first.nm
# RUN: cmp %t.winner.nm %t.large-first.nm
# RUN: FileCheck %s < %t.winner.nm

# CHECK: [[FOLD:[0-9a-f]+]] T fold_a
# CHECK-NEXT: [[FOLD]] T fold_b
# CHECK: T ordered_b
# CHECK-NOT: loser_only

#--- small.s
        .section .text$leader,"xr",largest,leader
        .globl leader
leader:
        retq
        .globl loser_only
loser_only:
        retq

        .addrsig
        .addrsig_sym loser_only
        .cg_profile loser_only, ordered_b, 1000000

#--- large.s
        .section .text$leader,"xr",largest,leader
        .globl leader
leader:
        .space 16, 0x90
        retq

#--- root.s
        .text
        .globl entry
entry:
        callq leader
        callq fold_a
        callq fold_b
        callq ordered_b
        retq

        .addrsig

        .section .text,"xr",one_only,fold_a
        .globl fold_a
fold_a:
        nop
        retq

        .section .text,"xr",one_only,fold_b
        .globl fold_b
fold_b:
        nop
        retq

        .section .text,"xr",one_only,ordered_b
        .globl ordered_b
ordered_b:
        nop
        nop
        retq
