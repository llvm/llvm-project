# Verify whole-group LARGEST replacement on i386, including SafeSEH metadata.
# ARM, ARM64, ARM64EC, and ARM64X have target-specific companion tests.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -triple i686-windows-msvc -filetype=obj \
# RUN:   %t.dir/small.s -o %t.i386-small.obj
# RUN: llvm-mc -triple i686-windows-msvc -filetype=obj \
# RUN:   %t.dir/large.s -o %t.i386-large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /safeseh /include:leader \
# RUN:   %t.i386-small.obj %t.i386-large.obj /out:%t.i386.dll
# RUN: llvm-objdump -s %t.i386.dll | FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# LARGE: Contents of section .text:
# LARGE: 44444444
# LARGE: Contents of section .rdata:
# LARGE: 55555555 bbbbbbbb

#--- small.s
        .def @feat.00;
        .scl 3;
        .type 0;
        .endef
        .globl @feat.00
@feat.00 = 1
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
        .section .rdata$assoc, "dr", associative, leader
        .long 0xaaaaaaaa
        .safeseh leader

#--- large.s
        .def @feat.00;
        .scl 3;
        .type 0;
        .endef
        .globl @feat.00
@feat.00 = 1
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .space 28, 0x44
        .section .rdata$assoc, "dr", associative, leader
        .long 0x55555555
        .long 0xbbbbbbbb
        .safeseh leader
