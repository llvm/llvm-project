# Verify that ADDR32NB relocs whose target lands in the .tls output
# section are tagged XFIXUP_TLS (0x2000). The classifier keys off the
# target output-section name, not the relocation type.

# REQUIRES: x86
# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.obj
# RUN: rm -f %t.pdb %t.dll
# RUN: lld-link /debug /debugtype:cv,fixup /pdb:%t.pdb /dll /out:%t.dll \
# RUN:   /entry:main /nodefaultlib %t.obj
# RUN: llvm-readobj --section-headers %t.dll > %t.combined
# RUN: llvm-pdbutil dump --fixups %t.pdb >> %t.combined
# RUN: FileCheck %s < %t.combined
# RUN: lld-link /debug /debugtype:cv,fixup /pdb:%t.pdb /dll /out:%t.dll \
# RUN:   /entry:main /nodefaultlib /merge:.tls=.rdata %t.obj
# RUN: llvm-readobj --section-headers %t.dll > %t.merged
# RUN: llvm-pdbutil dump --fixups %t.pdb >> %t.merged
# RUN: FileCheck %s --check-prefix=MERGED < %t.merged

# Capture .text and .tls section RVAs. main is the only thing in .text
# and gTlsVar is the only thing in .tls, so symbol RVA == section RVA.
# CHECK:      Name: .text
# CHECK:      VirtualAddress: 0x[[#%X, TEXT_RVA:]]
# CHECK:      Name: .tls
# CHECK:      VirtualAddress: 0x[[#%X, TLS_RVA:]]

# After the CodeView fixup: a function ptr to main (FUNCTION) and a
# TLS-data ptr (TLS) pointing at gTlsVar in .tls.
# CHECK:      Records: 3
# CHECK-NEXT: RVA
# CHECK-NEXT: | .rdata {{.*}} | .rdata | 3 | 0x0000 | DATA
# CHECK-NEXT: | .data        | [[#%.8X, TEXT_RVA]]  | .text        |    3 | 0x8000 | FUNCTION
# CHECK-NEXT: | .data        | [[#%.8X, TLS_RVA]]  | .tls         |    3 | 0x2000 | TLS

# Fixup classification follows the final output section: the merged TLS
# template in .rdata is DATA rather than TLS.
# MERGED: Name: .rdata
# MERGED-NOT: Name: .tls
# MERGED: Records: 3
# MERGED: | .data {{.*}} | .rdata | 3 | 0x0000 | DATA

        .text
        .def    main; .scl 2; .type 32; .endef
        .globl  main
main:
        movl    $0, %eax
        retq

        .section .tls,"dw"
        .globl  gTlsVar
gTlsVar:
        .long   0

        .data
        .globl  pMain
pMain:
        .long   main@IMGREL
        .globl  pTls
pTls:
        .long   gTlsVar@IMGREL
