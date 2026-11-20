# REQUIRES: x86

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/provider.s -o %t.provider.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/winning-reference.s -o %t.winning-reference.obj
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj \
# RUN:   %t.dir/losing-reference.s -o %t.losing-reference.obj
# RUN: lld-link /dll /nodefaultlib /entry:DllMainCRTStartup \
# RUN:   /export:variable,data /out:%t.provider.dll /implib:%t.provider.lib \
# RUN:   %t.provider.obj

# References from the prevailing LARGEST group must be visible to MinGW's
# automatic-import discovery.
# RUN: lld-link /lldmingw /nodefaultlib /entry:entry /subsystem:console \
# RUN:   /opt:noref /out:%t.winning.exe %t.small.obj \
# RUN:   %t.winning-reference.obj %t.root.obj %t.provider.lib
# RUN: llvm-readobj --coff-imports %t.winning.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT

# Conversely, a reference originating only in the losing group must not load
# an otherwise-unused import, including when section GC is disabled.
# RUN: lld-link /lldmingw /nodefaultlib /entry:entry /subsystem:console \
# RUN:   /opt:noref /out:%t.losing.exe %t.losing-reference.obj %t.large.obj \
# RUN:   %t.root.obj %t.provider.lib
# RUN: llvm-readobj --coff-imports %t.losing.exe | \
# RUN:   FileCheck %s --check-prefix=NO-IMPORT

# IMPORT: Import {
# IMPORT: Symbol: variable (0)
# NO-IMPORT-NOT: Import {

#--- provider.s
        .globl variable
        .globl DllMainCRTStartup
        .text
DllMainCRTStartup:
        retq
        .data
variable:
        .long 42

#--- root.s
        .text
        .globl entry
entry:
        leaq leader(%rip), %rax
        retq
        .globl _pei386_runtime_relocator
_pei386_runtime_relocator:
        retq

#--- small.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        retq

#--- large.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

#--- winning-reference.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        movl variable(%rip), %eax
        retq
        .space 32, 0x55

#--- losing-reference.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        movl variable(%rip), %eax
        retq
