# REQUIRES: x86

# Test a chain of associative COMDAT sections attached to competing
# IMAGE_COMDAT_SELECT_LARGEST groups.
#
# level1 is associated with leader, while level2 is associated with level1.
# Symbols defined in both associative levels must follow the prevailing leader.
# Superseding the smaller group must discard the full association chain.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.ref.small-large.map /out:%t.ref.small-large.exe
# RUN: llvm-objdump -s %t.ref.small-large.exe | FileCheck --check-prefix=IMAGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=22222222 \
# RUN:   --implicit-check-not=33333333 %s
# RUN: FileCheck --check-prefix=MAP %s < %t.ref.small-large.map

# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.noref.small-large.map /out:%t.noref.small-large.exe
# RUN: llvm-objdump -s %t.noref.small-large.exe | \
# RUN:   FileCheck --check-prefix=IMAGE --implicit-check-not=11111111 \
# RUN:     --implicit-check-not=22222222 --implicit-check-not=33333333 %s
# RUN: FileCheck --check-prefix=MAP %s < %t.noref.small-large.map

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /map:%t.ref.large-small.map /out:%t.ref.large-small.exe
# RUN: llvm-objdump -s %t.ref.large-small.exe | FileCheck --check-prefix=IMAGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=22222222 \
# RUN:   --implicit-check-not=33333333 %s
# RUN: FileCheck --check-prefix=MAP %s < %t.ref.large-small.map

# IMAGE-DAG: 44444444
# IMAGE-DAG: 55555555
# IMAGE-DAG: 66666666

# MAP-DAG: leader{{.*}}large.obj
# MAP-DAG: assoc_level1{{.*}}large.obj
# MAP-DAG: nested_public{{.*}}large.obj

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111

        .section .rdata$level1, "dr", associative, leader
        .globl assoc_level1
assoc_level1:
        .long 0x22222222

        .section .rdata$level2, "dr", associative, assoc_level1
        .globl nested_public
nested_public:
        .long 0x33333333

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .long 0x45454545
        .long 0x46464646

        .section .rdata$level1, "dr", associative, leader
        .globl assoc_level1
assoc_level1:
        .long 0x55555555

        .section .rdata$level2, "dr", associative, assoc_level1
        .globl nested_public
nested_public:
        .long 0x66666666

#--- root.s
        .section .text$root, "xr"
        .globl entry
entry:
        leaq leader(%rip), %rax
        leaq assoc_level1(%rip), %rcx
        leaq nested_public(%rip), %rdx
        retq
