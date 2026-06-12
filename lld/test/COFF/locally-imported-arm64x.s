// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t.arm64ec.obj

// RUN: lld-link -machine:arm64x -dll -noentry %t.arm64.obj %t.arm64ec.obj -out:%t.dll 2>&1 | FileCheck --check-prefix=WARN %s
// WARN:      lld-link: warning: {{.*}}.arm64.obj: locally defined symbol imported: func (native symbol)
// WARN-NEXT: lld-link: warning: {{.*}}.arm64ec.obj: locally defined symbol imported: func (EC symbol)

// RUN: llvm-readobj --hex-dump=.test %t.dll | FileCheck --check-prefix=TEST %s
// TEST: 0x180005000 00300000 08300000

// RUN: llvm-readobj --coff-basereloc %t.dll | FileCheck --check-prefix=RELOCS %s
// RELOCS:      Entry {
// RELOCS-NEXT:   Type: DIR64
// RELOCS-NEXT:   Address: 0x3000
// RELOCS-NEXT: }
// RELOCS-NEXT: Entry {
// RELOCS-NEXT:   Type: DIR64
// RELOCS-NEXT:   Address: 0x3008
// RELOCS-NEXT: }

// RUN: llvm-readobj --hex-dump=.rdata %t.dll | FileCheck --check-prefix=RDATA %s
// RDATA: 0x180003000 00100080 01000000 00200080 01000000

    .text
    .globl func
func:
    ret

    .section .test, "r"
    .rva __imp_func
