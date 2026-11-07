# REQUIRES: x86

# Test successive IMAGE_COMDAT_SELECT_LARGEST replacements. In the primary
# order, the small group is replaced by the medium group, which is then
# replaced by the large group. Secondary symbols and symbols in associative
# sections must follow the final prevailing candidate.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/medium.s -o %t.medium.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# Two successive replacements.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.medium.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.ref.sml.map /out:%t.ref.sml.exe
# RUN: llvm-objdump -s %t.ref.sml.exe | FileCheck --check-prefix=IMAGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=12121212 \
# RUN:   --implicit-check-not=aaaaaaaa --implicit-check-not=22222222 \
# RUN:   --implicit-check-not=23232323 --implicit-check-not=bbbbbbbb %s
# RUN: FileCheck --check-prefix=MAP %s < %t.ref.sml.map

# RUN: lld-link /opt:noref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.medium.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.noref.sml.map /out:%t.noref.sml.exe
# RUN: llvm-objdump -s %t.noref.sml.exe | FileCheck --check-prefix=IMAGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=12121212 \
# RUN:   --implicit-check-not=aaaaaaaa --implicit-check-not=22222222 \
# RUN:   --implicit-check-not=23232323 --implicit-check-not=bbbbbbbb %s
# RUN: FileCheck --check-prefix=MAP %s < %t.noref.sml.map

# Largest candidate first: no later candidate may replace it.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.medium.obj %t.small.obj %t.root.obj \
# RUN:   /map:%t.ref.lms.map /out:%t.ref.lms.exe
# RUN: llvm-objdump -s %t.ref.lms.exe | FileCheck --check-prefix=IMAGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=12121212 \
# RUN:   --implicit-check-not=aaaaaaaa --implicit-check-not=22222222 \
# RUN:   --implicit-check-not=23232323 --implicit-check-not=bbbbbbbb %s
# RUN: FileCheck --check-prefix=MAP %s < %t.ref.lms.map

# IMAGE-DAG: 44444444
# IMAGE-DAG: 55555555
# IMAGE-DAG: cccccccc

# MAP-DAG: leader{{.*}}large.obj
# MAP-DAG: secondary{{.*}}large.obj
# MAP-DAG: assoc_public{{.*}}large.obj

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
        .globl secondary
secondary:
        .long 0x12121212

        .section .rdata$assoc, "dr", associative, leader
        .globl assoc_public
assoc_public:
        .long 0xaaaaaaaa

#--- medium.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x22222222
        .long 0x22222223
        .globl secondary
secondary:
        .long 0x23232323

        .section .rdata$assoc, "dr", associative, leader
        .globl assoc_public
assoc_public:
        .long 0xbbbbbbbb

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .long 0x45454545
        .long 0x46464646
        .globl secondary
secondary:
        .long 0x55555555

        .section .rdata$assoc, "dr", associative, leader
        .globl assoc_public
assoc_public:
        .long 0xcccccccc

#--- root.s
        .section .text$root, "xr"
        .globl entry
entry:
        leaq leader(%rip), %rax
        leaq secondary(%rip), %rcx
        leaq assoc_public(%rip), %rdx
        retq
