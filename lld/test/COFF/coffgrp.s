# REQUIRES: x86

# Make a DLL that exports exportfn1.
# RUN: yaml2obj %p/Inputs/export.yaml -o %t.obj
# RUN: lld-link /out:%t.dll /dll %t.obj /export:exportfn1 /implib:%t.lib

# Make an obj that takes the address of that exported function.
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t2.obj
# RUN: lld-link -entry:main -guard:cf %t2.obj %t.lib -nodefaultlib -out:%t.exe /verbose /coff-debug-record 2>&1 | FileCheck %s --check-prefix=NOPGO,REC
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -defsym HOT=1 %s -o %t2h.obj
# RUN: lld-link -entry:main -guard:cf %t2h.obj %t.lib -nodefaultlib -out:%t.exe /verbose /coff-debug-record 2>&1 | FileCheck %s --check-prefix=PGO,REC

# NOPGO:      coffgrp signature: ''
# NOPGO-NEXT: coffgrp rec .text$abcdef RVA = 0x1000 size = 0x08
# PGO:        coffgrp signature: 'PGO'
# PGO-NEXT:   coffgrp rec .text$_00hot RVA = 0x1000 size = 0x08
# REC:        coffgrp rec .text RVA = 0x1010 size = 0x06
# REC-NEXT:   coffgrp rec .rdata RVA = 0x2000 size = 0x114
# REC-NEXT:   coffgrp rec .idata$2 RVA = 0x21d8 size = 0x28
# REC-NEXT:   coffgrp rec .idata$4 RVA = 0x2200 size = 0x10
# REC-NEXT:   coffgrp rec .idata$5 RVA = 0x2210 size = 0x10
# REC-NEXT:   coffgrp rec .idata$6 RVA = 0x2220 size = 0x0c
# REC-NEXT:   coffgrp rec .idata$7 RVA = 0x222c size = 0x12

        .def     @feat.00;
        .scl    3;
        .type   0;
        .endef
        .globl  @feat.00
@feat.00 = 0x001

.ifndef HOT
	.section .text$abcdef,"xr"
.else
	.section .text$_00hot,"xr"
.endif
        .def     main; .scl    2; .type   32; .endef
        .global main
main:
        leaq exportfn1(%rip), %rax
        retq

        .section .rdata,"dr"
.globl _load_config_used
_load_config_used:
        .long 256
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 128, 1, 0

