# Verify that section-relative and section-index relocations do not produce
# PEFIXUP records. Even when no input relocation contributes a fixup,
# the CodeView debug-directory pointer must still have one.

# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: rm -f %t.pdb %t.dll
# RUN: lld-link /debug /debugtype:cv,fixup /pdb:%t.pdb /dll /out:%t.dll \
# RUN:   /entry:main /nodefaultlib %t.obj
# RUN: llvm-readobj --section-headers %t.dll > %t.combined
# RUN: llvm-pdbutil dump --fixups %t.pdb >> %t.combined
# RUN: FileCheck %s < %t.combined

# RUN: llvm-mc -triple x86_64-windows-msvc -defsym DEBUG_ONLY=1 %s -filetype=obj -o %t.empty.obj
# RUN: lld-link /debug /debugtype:cv,fixup /brepro /cetcompat /WX /pdb:%t.empty.pdb /dll /out:%t.empty.dll \
# RUN:   /entry:main /nodefaultlib %t.empty.obj
# RUN: llvm-readobj --file-headers --coff-debug-directory %t.empty.dll > %t.empty.txt
# RUN: llvm-pdbutil dump --streams --fixups %t.empty.pdb >> %t.empty.txt
# RUN: FileCheck %s --check-prefix=CVONLY < %t.empty.txt

# The first debug directory entry is CodeView. Only its AddressOfRawData
# field (at offset 0x14) gets a fixup; there is none for PointerToRawData
# or for the other directory entries.
# CVONLY: DebugRVA: 0x[[#%X,DEBUG:]]
# CVONLY: Type: CodeView
# CVONLY: AddressOfRawData: 0x[[#%X,CV:]]
# CVONLY: Type: ExtendedDLLCharacteristics
# CVONLY: Type: Repro
# CVONLY: ({{ *}}12 bytes): [Fixup Data]
# CVONLY: Records: 1
# CVONLY-NEXT: RVA
# CVONLY-NEXT: [[#%.8X,DEBUG + 0x14]] | .rdata | [[#%.8X,CV]] | .rdata | 3 | 0x0000 | DATA

# Capture .text and .rdata section RVAs. main is the only thing in .text
# and gMessage is the only thing in .rdata, so symbol RVA == section RVA.
# CHECK:      Name: .text
# CHECK:      VirtualAddress: 0x[[#%X, TEXT_RVA:]]
# CHECK:      Name: .rdata
# CHECK:      VirtualAddress: 0x[[#%X, RDATA_RVA:]]

# The .data section has two ADDR32NB relocs and a SECREL/SECTION pair.
# Only the two ADDR32NB-derived records should appear after the CodeView fixup.
# CHECK:      Records: 3
# CHECK-NEXT: RVA
# CHECK-NEXT: | .rdata {{.*}} | .rdata | 3 | 0x0000 | DATA
# CHECK-NEXT: | .data        | [[#%.8X, TEXT_RVA]]  | .text        |    3 | 0x8000 | FUNCTION
# CHECK-NEXT: | .data        | [[#%.8X, RDATA_RVA]]  | .rdata       |    3 | 0x0000 | DATA
# CHECK-NOT:  | .debug

        .cv_file 1 "test.s"
        .text
        .def    main; .scl 2; .type 32; .endef
        .globl  main
main:
        .cv_func_id 0
        .cv_loc 0 1 1 0
        movl    $0, %eax
        retq
.Lmain_end:

        .ifndef DEBUG_ONLY
        .section .rdata,"dr"
gMessage:
        .asciz  "hi"

        .data
        .globl  pMain
pMain:
        .long   main@IMGREL
        .globl  pData
pData:
        .long   gMessage@IMGREL
        .endif

        .data
        .secrel32 main
        .secidx main

        .section .debug$S,"dr"
        .long   4
        .cv_linetable 0, main, .Lmain_end
        .cv_filechecksums
        .cv_stringtable
