# REQUIRES: x86

# RUN: llvm-mc -triple x86_64-windows-msvc -filetype=obj %s -o %t.obj
# RUN: lld-link %t.obj -out:%t.dll -dll -noentry -nodefaultlib -dependentloadflag:0x800 2>&1 | FileCheck %s
# CHECK: lld-link: warning: '_load_config_used' structure too small to include DependentLoadFlags

        .section .rdata,"dr"
        .balign 8
.globl _load_config_used
_load_config_used:
        .long 0x4c
        .fill 0x48, 1, 0
