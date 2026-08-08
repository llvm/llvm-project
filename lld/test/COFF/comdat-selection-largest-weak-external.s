# REQUIRES: x86

# Test that a weak external keeps its fallback when its initially prevailing
# definition belongs to an IMAGE_COMDAT_SELECT_LARGEST group that is later
# superseded.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/medium.s -o %t.medium.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/weak-reference.s -o %t.weak-reference.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/weak-reference-2.s -o %t.weak-reference-2.obj

# Weak aliases encountered while a regular definition prevails used to be
# ignored. Deferring an alias in case that definition is later superseded must
# not make a second weak alias diagnose the prevailing definition as a
# duplicate.
# RUN: lld-link /dll /noentry /nodefaultlib %t.small.obj \
# RUN:   %t.weak-reference.obj %t.weak-reference-2.obj \
# RUN:   /out:%t.multiple-weak-while-defined.dll

# If the provisional definition is later discarded, replay all suppressed
# aliases and preserve the normal conflicting-alias diagnostic.
# RUN: not lld-link /dll /noentry /nodefaultlib %t.small.obj \
# RUN:   %t.weak-reference.obj %t.weak-reference-2.obj %t.large.obj \
# RUN:   /out:%t.multiple-weak-after-replacement.dll 2>&1 | \
# RUN:   FileCheck --check-prefix=WEAK-CONFLICT %s
# WEAK-CONFLICT: duplicate symbol: foo_lazy

# The weak external is first observed while foo_lazy is defined. Its fallback
# must be deferred until the smaller group is superseded.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.weak-reference.obj %t.large.obj \
# RUN:   /out:%t.definition-first.exe
# RUN: llvm-objdump -s %t.definition-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

# Preserve the fallback as the symbol changes from undefined to defined and
# back to undefined.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.weak-reference.obj %t.small.obj %t.large.obj \
# RUN:   /out:%t.weak-first.exe
# RUN: llvm-objdump -s %t.weak-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

# Keep the fallback across more than one temporary definition.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.weak-reference.obj %t.small.obj %t.medium.obj %t.large.obj \
# RUN:   /out:%t.replacement-chain.exe
# RUN: llvm-objdump -s %t.replacement-chain.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 \
# RUN:     --implicit-check-not=22222222 %s

# Control cases in which the final largest group is already known before the
# weak external is read.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.weak-reference.obj \
# RUN:   /out:%t.largest-before-weak.exe
# RUN: llvm-objdump -s %t.largest-before-weak.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.small.obj %t.weak-reference.obj \
# RUN:   /out:%t.largest-first.exe
# RUN: llvm-objdump -s %t.largest-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 %s

# IMAGE: Contents of section .text:
# IMAGE-DAG: 77777777
# IMAGE-DAG: 44444444

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 8, 0x11

        .globl foo_lazy
foo_lazy:
        .space 8, 0x11

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

#--- medium.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 12, 0x22

        .globl foo_lazy
foo_lazy:
        .space 12, 0x22

#--- weak-reference.s
        .section .text$fallback, "xr"
        .globl fallback
fallback:
        .space 8, 0x77

        .weak foo_lazy
        foo_lazy = fallback

        .section .text$entry, "xr"
        .globl entry
entry:
        leaq foo_lazy(%rip), %rax
        leaq leader(%rip), %rcx
        retq

#--- weak-reference-2.s
        .section .text$fallback2, "xr"
        .globl fallback2
fallback2:
        .space 8, 0x88

        .weak foo_lazy
        foo_lazy = fallback2
