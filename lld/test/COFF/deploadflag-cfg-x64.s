# REQUIRES: x86

# RUN: llvm-mc -triple x86_64-windows-msvc -filetype=obj %s -o %t.ldcfg.obj

# RUN: lld-link %S/Inputs/precomp-a.obj %t.ldcfg.obj /out:%t.exe /nodefaultlib /force
# RUN: llvm-readobj --coff-load-config %t.exe | FileCheck %s --check-prefix FLAGS-400

# RUN: lld-link %S/Inputs/precomp-a.obj %t.ldcfg.obj /out:%t.exe /nodefaultlib /force /dependentloadflag:0x800
# RUN: llvm-readobj --coff-load-config %t.exe | FileCheck %s --check-prefix FLAGS-800

# MSVC linker does not rewrite non-zero value of dependentloadflag in _load_config_used with zero
# RUN: lld-link %S/Inputs/precomp-a.obj %t.ldcfg.obj /out:%t.exe /nodefaultlib /force /dependentloadflag:0x0
# RUN: llvm-readobj --coff-load-config %t.exe | FileCheck %s --check-prefix FLAGS-400

# RUN: lld-link %S/Inputs/precomp-a.obj %t.ldcfg.obj /out:%t.exe /nodefaultlib /force /dependentloadflag:0x800 /dependentloadflag:0x0
# RUN: llvm-readobj --coff-load-config %t.exe | FileCheck %s --check-prefix FLAGS-400

# FLAGS-800: DependentLoadFlags: 0x800
# FLAGS-400: DependentLoadFlags: 0x400

        .section .rdata,"dr"
.globl _load_config_used
_load_config_used:
        .long 256
        .fill 74, 1, 0
        .byte 0x00
        .byte 0x40
        .fill 48, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 128, 1, 0

