# Verify that synthetic chunks emit their own PEFIXUP records via
# /guard:cf, which materializes the GFIDS RVATableChunk in .rdata.

# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: rm -f %t.pdb %t.dll
# RUN: lld-link /debug /debugtype:cv,fixup /guard:cf,nolongjmp /pdb:%t.pdb \
# RUN:   /dll /out:%t.dll /entry:main /nodefaultlib %t.obj
# RUN: llvm-readobj --file-headers --coff-load-config %t.dll > %t.combined
# RUN: llvm-pdbutil dump --fixups %t.pdb >> %t.combined
# RUN: FileCheck %s < %t.combined

# Capture the image base and the GuardFidTable entries. Entries are sorted
# by RVA, and the input layout puts main, then f, then g (declaration
# order via .section .text,one_only,<name>), so the sort is deterministic.
# CHECK:      ImageBase: 0x[[#%X, BASE:]]
# CHECK:      GuardFidTable [
# CHECK-NEXT:   0x[[#%X, MAIN_VA:]]
# CHECK-NEXT:   0x[[#%X, F_VA:]]
# CHECK-NEXT:   0x[[#%X, G_VA:]]

# The synthetic GFIDS chunk in .rdata contributes one FUNCTION-tagged
# ADDR32NB record per address-taken entry. The records are sorted by
# source RVA (the .gfids$y slot's location), which mirrors the table
# layout above: main, f, g. Validate each TargetRVA points at the right
# function by computing it from the captured GuardFidTable VAs.
# CHECK:      Fixup Data
# CHECK:      [[#%X, TABLE_RVA:]] | .rdata | [[#%.8X, MAIN_VA - BASE]] | .text | 3 | 0x8000 | FUNCTION
# CHECK-NEXT: [[#%.8X, TABLE_RVA + 4]] | .rdata | [[#%.8X, F_VA - BASE]] | .text | 3 | 0x8000 | FUNCTION
# CHECK-NEXT: [[#%.8X, TABLE_RVA + 8]] | .rdata | [[#%.8X, G_VA - BASE]] | .text | 3 | 0x8000 | FUNCTION

        .def     @feat.00; .scl 3; .type 0; .endef
        .globl  @feat.00
@feat.00 = 0x801

        .def    main; .scl 2; .type 32; .endef
        .section .text,"xr",one_only,main
        .globl  main
main:
        callq   f
        callq   g
        retq

        .def    f; .scl 2; .type 32; .endef
        .section .text,"xr",one_only,f
        .globl  f
f:
        retq

        .def    g; .scl 2; .type 32; .endef
        .section .text,"xr",one_only,g
        .globl  g
g:
        retq

        .data
        .globl  fp_main
fp_main:
        .quad   main
        .globl  fp_f
fp_f:
        .quad   f
        .globl  fp_g
fp_g:
        .quad   g

        .section .gfids$y,"dr",associative,fp_main
        .symidx main
        .section .gfids$y,"dr",associative,fp_f
        .symidx f
        .section .gfids$y,"dr",associative,fp_g
        .symidx g

        .section .rdata,"dr"
        .p2align 3
        .globl _load_config_used
_load_config_used:
        .long 256
        .fill 124, 1, 0
        .quad __guard_fids_table
        .quad __guard_fids_count
        .long __guard_flags
        .fill 128, 1, 0
