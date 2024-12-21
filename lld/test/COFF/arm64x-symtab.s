// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym.s -o sym-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym.s -o sym-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows sym.s -o sym-x86_64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows symref.s -o symref-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows symref.s -o symref-arm64ec.obj
// RUN: llvm-lib -machine:arm64x -out:sym.lib sym-aarch64.obj sym-arm64ec.obj

// Check that native object files can't reference EC symbols.

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err1.dll symref-aarch64.obj sym-arm64ec.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEF %s
// UNDEF:      lld-link: error: undefined symbol: sym
// UNDEF-NEXT: >>> referenced by symref-aarch64.obj:(.data)

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err2.dll symref-aarch64.obj sym-x86_64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEF %s

// Check that ARM64X target can have the same symbol names in both native and EC namespaces.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out.dll symref-aarch64.obj sym-aarch64.obj \
// RUN:           symref-arm64ec.obj sym-x86_64.obj

// Check that ARM64X target can reference both native and EC symbols from an archive.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out2.dll symref-aarch64.obj symref-arm64ec.obj sym.lib

#--- symref.s
    .data
    .rva sym

#--- sym.s
     .data
     .globl sym
sym:
     .word 0
