# REQUIRES: x86
#
# ANY and LARGEST are merged as LARGEST for MSVC compatibility. Secondary
# definitions must follow the same group decision and suppressed lazy providers
# must be replayed when the group that hid them loses.
#
# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/any-small.s -o %t.any-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/any-large.s -o %t.any-large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/largest-small.s -o %t.largest-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/largest-large.s -o %t.largest-large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/same-size-small.s -o %t.same-size-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider.s -o %t.provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-lib -machine:amd64 -out:%t.provider.lib %t.provider.obj
#
# ANY first, larger LARGEST second.
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.any-small.obj %t.provider.lib %t.largest-large.obj %t.root.obj \
# RUN:   /map:%t.any-largest.map /out:%t.any-largest.exe
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.any-largest.map
# RUN: llvm-objdump -s %t.any-largest.exe | FileCheck %s --check-prefix=LARGE
#
# LARGEST first, larger ANY second. The compatibility merge must still use
# size, not input order.
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.largest-small.obj %t.provider.lib %t.any-large.obj %t.root.obj \
# RUN:   /map:%t.largest-any.map /out:%t.largest-any.exe
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.largest-any.map
# RUN: llvm-objdump -s %t.largest-any.exe | FileCheck %s --check-prefix=LARGE
#
# The ANY/LARGEST promotion must persist for later candidates. Without that,
# the third ANY candidate is incorrectly compared as ANY/ANY and loses.
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.any-small.obj %t.largest-small.obj %t.any-large.obj \
# RUN:   %t.provider.lib %t.root.obj \
# RUN:   /out:%t.any-largest-any.exe
# RUN: llvm-objdump -s %t.any-largest-any.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE
#
# MinGW's ANY/SAME_SIZE compatibility promotion is likewise group state. The
# later, differently-sized ANY candidate must diagnose a size mismatch.
# RUN: not lld-link /lldmingw /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.any-small.obj %t.same-size-small.obj %t.any-large.obj \
# RUN:   %t.provider.lib %t.root.obj \
# RUN:   /out:%t.any-same-any.exe 2>&1 | FileCheck %s --check-prefix=SAME-SIZE
#
# PROVIDER: only_loser{{.*}}provider.obj
# LARGE: 44444444
# LARGE-NOT: 11111111
# SAME-SIZE: duplicate symbol: leader
#
#--- any-small.s
        .section .text$mix, "xr", discard, leader
        .globl leader
leader:
        .byte 0x11
        .globl only_loser
only_loser:
        .long 0x11111111
#
#--- any-large.s
        .section .text$mix, "xr", discard, leader
        .globl leader
leader:
        .space 32, 0x44
#
#--- largest-small.s
        .section .text$mix, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .globl only_loser
only_loser:
        .long 0x11111111
#
#--- largest-large.s
        .section .text$mix, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44
#
#--- same-size-small.s
        .section .text$mix, "xr", same_size, leader
        .globl leader
leader:
        .byte 0x11
        .globl only_loser
only_loser:
        .long 0x11111111
#
#--- provider.s
        .section .rdata$provider, "dr"
        .globl only_loser
only_loser:
        .long 0x77777777
#
#--- root.s
        .text
        .globl entry
entry:
        leaq leader(%rip), %rax
        movl only_loser(%rip), %ecx
        retq
