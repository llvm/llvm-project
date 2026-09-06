# REQUIRES: aarch64

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/small.s -o %t.arm64-small.obj
# RUN: llvm-mc -triple aarch64-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/large.s -o %t.arm64-large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /include:leader \
# RUN:   %t.arm64-small.obj %t.arm64-large.obj /out:%t.arm64.dll
# RUN: llvm-objdump -s %t.arm64.dll | FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/small.s -o %t.ec-small.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/large.s -o %t.ec-large.obj
# RUN: lld-link /dll /noentry /nodefaultlib /machine:arm64ec /include:leader \
# RUN:   %t.ec-small.obj %t.ec-large.obj /out:%t.ec.dll
# RUN: llvm-objdump -s %t.ec.dll | FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/small-alias.s -o %t.ec-small-alias.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/alias-ref.s -o %t.ec-alias-ref.obj
# RUN: llvm-mc -triple arm64ec-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/alias-provider.s -o %t.ec-alias-provider.obj
# RUN: llvm-mc -triple x86_64-pc-windows-msvc -filetype=obj \
# RUN:   %t.dir/x64-ref.s -o %t.x64-ref.obj
# RUN: llvm-lib -machine:arm64ec -out:%t.ec-alias-provider.lib \
# RUN:   %t.ec-alias-provider.obj
# RUN: lld-link /dll /noentry /nodefaultlib /machine:arm64ec /include:func \
# RUN:   %t.ec-small-alias.obj %t.ec-alias-ref.obj %t.ec-alias-provider.obj \
# RUN:   %t.ec-large.obj /out:%t.ec-alias.dll
# RUN: llvm-objdump -s %t.ec-alias.dll | FileCheck %s --check-prefix=ALIAS \
# RUN:   --implicit-check-not=22222222
# RUN: lld-link /dll /noentry /nodefaultlib /machine:arm64x /include:func \
# RUN:   %t.ec-small-alias.obj %t.ec-alias-ref.obj %t.ec-alias-provider.obj \
# RUN:   %t.ec-large.obj /out:%t.arm64x-alias.dll
# RUN: llvm-objdump -s %t.arm64x-alias.dll | \
# RUN:   FileCheck %s --check-prefix=ALIAS --implicit-check-not=22222222
# RUN: lld-link /dll /noentry /nodefaultlib /machine:arm64ec /include:func \
# RUN:   %t.ec-small-alias.obj %t.x64-ref.obj %t.ec-alias-provider.lib \
# RUN:   %t.ec-large.obj /out:%t.ec-x64-alias.dll
# RUN: llvm-objdump -s %t.ec-x64-alias.dll | FileCheck %s \
# RUN:   --check-prefix=ALIAS --implicit-check-not=22222222
# RUN: lld-link /dll /noentry /nodefaultlib /machine:arm64x /include:leader \
# RUN:   %t.arm64-small.obj %t.arm64-large.obj %t.ec-small.obj %t.ec-large.obj \
# RUN:   /out:%t.arm64x.dll
# RUN: llvm-objdump -s %t.arm64x.dll | FileCheck %s --check-prefix=LARGE \
# RUN:   --implicit-check-not=11111111 --implicit-check-not=aaaaaaaa

# LARGE: Contents of section .text:
# LARGE: 44444444
# LARGE: Contents of section .rdata:
# LARGE: 55555555 bbbbbbbb
# ALIAS: 77777777

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
        .section .rdata$assoc, "dr", associative, leader
        .long 0xaaaaaaaa

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x44444444
        .space 28, 0x44
        .section .rdata$assoc, "dr", associative, leader
        .long 0x55555555
        .long 0xbbbbbbbb

#--- small-alias.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long 0x11111111
        .globl func
func:
        .long 0x22222222

#--- alias-ref.s
        .weak_anti_dep func
        .set func, "#func"

#--- alias-provider.s
        .text
        .globl "#func"
"#func":
        .long 0x77777777

#--- x64-ref.s
        .data
        .rva func
