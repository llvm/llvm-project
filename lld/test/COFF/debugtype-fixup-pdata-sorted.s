# Verify that PEFIXUP records describe .pdata as it appears in the image.
# lld sorts .pdata in place in the output buffer after chunk layout is final,
# so fixup site RVAs derived from chunk layout must be remapped onto the
# sorted entries.

# REQUIRES: aarch64
# RUN: llvm-mc -triple aarch64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: rm -f %t.pdb %t.dll
# RUN: lld-link /debug /debugtype:cv,fixup /pdb:%t.pdb /dll /out:%t.dll \
# RUN:   /entry:alpha /nodefaultlib /opt:noref %t.obj
# RUN: llvm-readobj --file-headers --section-headers --coff-debug-directory %t.dll > %t.txt
# RUN: llvm-pdbutil dump --fixups %t.pdb >> %t.txt
# RUN: FileCheck %s < %t.txt

# The object lists beta's RUNTIME_FUNCTION (packed unwind, so no .xdata
# reference) before alpha's (which points at .xdata), while alpha has the
# lower RVA. Sorting swaps the two entries, so the .xdata record belongs to
# the first entry and must appear between the two function records. Before
# the remap it was emitted last, against beta's old slot.

# CHECK:      DebugRVA: 0x[[#%X,DEBUG:]]
# CHECK:      Name: .text
# CHECK:      VirtualAddress: 0x[[#%X,TEXT:]]
# CHECK:      Name: .pdata
# CHECK:      VirtualAddress: 0x[[#%X,PDATA:]]
# CHECK:      Type: CodeView
# CHECK:      AddressOfRawData: 0x[[#%X,CV:]]
# CHECK:      Records: 4
# CHECK-NEXT: RVA
# CHECK-NEXT: [[#%.8X,DEBUG + 0x14]] | .rdata | [[#%.8X,CV]] | .rdata | 2 | 0x0000 | DATA
# CHECK-NEXT: [[#%.8X,PDATA]] | .pdata | [[#%.8X,TEXT]] | .text | 2 | 0x8000 | FUNCTION
# CHECK-NEXT: [[#%.8X,PDATA + 4]] | .pdata {{.*}} | .rdata | 2 | 0x0000 | DATA
# CHECK-NEXT: [[#%.8X,PDATA + 8]] | .pdata | [[#%.8X,TEXT + 4]] | .text | 2 | 0x8000 | FUNCTION

        .text
        .def    alpha; .scl 2; .type 32; .endef
        .globl  alpha
        .p2align 2
alpha:
        ret

        .def    beta; .scl 2; .type 32; .endef
        .globl  beta
        .p2align 2
beta:
        ret

        .section        .xdata,"dr"
        .p2align 2
alpha_xdata:
        .long   0x10400001

        .section        .pdata,"dr"
        .p2align 2
        .long   beta@IMGREL
        .long   0x00000001
        .long   alpha@IMGREL
        .long   alpha_xdata@IMGREL
