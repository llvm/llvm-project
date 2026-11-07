# REQUIRES: x86

# Verify that external symbols defined directly in associative sections follow
# the prevailing IMAGE_COMDAT_SELECT_LARGEST leader.
#
# The smaller and larger candidates both define assoc_public in an associative
# section. When the larger leader supersedes the smaller one, the old
# assoc_public definition must disappear with its section and the reference
# from root.obj must resolve to the larger candidate.
#
# Cover the important symbol-table states:
#
#   small, large, root: replacement happens before the reference;
#   small, root, large: the old definition is referenced before replacement;
#   root, small, large: an undefined reference exists before either definition;
#   large, small, root: the smaller candidate loses without becoming prevailing.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.small-large-root-ref.exe
# RUN: llvm-objdump -s %t.small-large-root-ref.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.small-large-root-noref.exe
# RUN: llvm-objdump -s %t.small-large-root-noref.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.large-small-root-ref.exe
# RUN: llvm-objdump -s %t.large-small-root-ref.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:noref \
# RUN:   %t.large.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.large-small-root-noref.exe
# RUN: llvm-objdump -s %t.large-small-root-noref.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.small.obj %t.root.obj %t.large.obj \
# RUN:   /out:%t.small-root-large.exe
# RUN: llvm-objdump -s %t.small-root-large.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.root.obj %t.small.obj %t.large.obj \
# RUN:   /out:%t.root-small-large.exe
# RUN: llvm-objdump -s %t.root-small-large.exe | \
# RUN:   FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# LARGE: Contents of section .rdata:
# LARGE-NEXT: {{[0-9a-f]+}} 55555555 bbbbbbbb

#--- small.s
        .section .text$largest, "xr", largest, leader

        .globl leader
leader:
        .byte 0x11

        .section .rdata$assoc, "dr", associative, leader

        .globl assoc_public
assoc_public:
        .long 0x11111111
        .long 0xaaaaaaaa

#--- large.s
        .section .text$largest, "xr", largest, leader

        .globl leader
leader:
        .space 32, 0x44

        .section .rdata$assoc, "dr", associative, leader

        .globl assoc_public
assoc_public:
        .long 0x55555555
        .long 0xbbbbbbbb

#--- root.s
        .section .text$root, "xr"

        .globl entry
entry:
        movl assoc_public(%rip), %eax
        retq
