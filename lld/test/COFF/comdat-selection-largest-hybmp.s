# REQUIRES: aarch64

# Hybrid-map entries are symbol-table side effects. A map in a losing LARGEST
# group must never install an entry thunk. Winner+loser in either order is
# therefore byte-identical to winner alone.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %S/Inputs/loadconfig-arm64ec.s -o %t.loadcfg.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/func.s -o %t.func.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj

# RUN: lld-link /machine:arm64ec /dll /noentry /nodefaultlib /opt:ref \
# RUN:   /timestamp:0 \
# RUN:   /include:func %t.loadcfg.obj %t.func.obj %t.large.obj \
# RUN:   /out:%t.winner.dll
# RUN: lld-link /machine:arm64ec /dll /noentry /nodefaultlib /opt:ref \
# RUN:   /timestamp:0 \
# RUN:   /include:func %t.loadcfg.obj %t.func.obj %t.small.obj %t.large.obj \
# RUN:   /out:%t.small-first.dll
# RUN: lld-link /machine:arm64ec /dll /noentry /nodefaultlib /opt:ref \
# RUN:   /timestamp:0 \
# RUN:   /include:func %t.loadcfg.obj %t.func.obj %t.large.obj %t.small.obj \
# RUN:   /out:%t.large-first.dll
# RUN: cmp %t.winner.dll %t.small-first.dll
# RUN: cmp %t.winner.dll %t.large-first.dll
# RUN: llvm-objdump -d %t.winner.dll | FileCheck %s

# CHECK: mov w0, #0x2
# CHECK-NOT: mov w0, #0x1

#--- func.s
        .section .text$func,"xr",discard,func
        .globl func
        .p2align 2
func:
        mov w0, #0
        ret

#--- small.s
        .section .wowthk$small,"xr",discard,bad_thunk
        .globl bad_thunk
        .p2align 2
bad_thunk:
        mov w0, #1
        ret

        .section .hybmp$x,"yi",largest,hybmp_group
        .globl hybmp_group
hybmp_group:
        .symidx func
        .symidx bad_thunk
        .word 1

#--- large.s
        .section .wowthk$large,"xr",discard,good_thunk
        .globl good_thunk
        .p2align 2
good_thunk:
        mov w0, #2
        ret

        .section .hybmp$x,"yi",largest,hybmp_group
        .globl hybmp_group
hybmp_group:
        .symidx func
        .symidx good_thunk
        .word 1
        .symidx func
        .symidx func
        .word 0
