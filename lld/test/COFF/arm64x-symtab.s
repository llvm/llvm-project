// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym.s -o sym-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym.s -o sym-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows sym.s -o sym-x86_64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows symref.s -o symref-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows symref.s -o symref-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=x86_64-windows symref.s -o symref-x86_64.obj
// RUN: llvm-as sym.ll -o sym.ll.obj
// RUN: llvm-lib -machine:arm64x -out:sym.lib sym-aarch64.obj sym-arm64ec.obj
// RUN: llvm-lib -machine:amd64 -out:sym-x86_64.lib sym-x86_64.obj
// RUN: llvm-lib -machine:amd64 -out:sym-ll.lib sym.ll.obj
// RUN: llvm-lib -machine:amd64 -out:sym-imp.lib -def:sym.def
// RUN: llvm-lib -machine:arm64 -out:sym-aarch64.lib sym-aarch64.obj

// Check that native object files can't reference EC symbols.

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err1.dll symref-aarch64.obj sym-arm64ec.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEF %s
// UNDEF:      lld-link: error: undefined symbol: sym
// UNDEF-NEXT: >>> referenced by symref-aarch64.obj:(.data)

// Check that EC object files can't reference native symbols.

// RUN: not lld-link -machine:arm64x -dll -noentry -out:out.dll symref-arm64ec.obj sym-aarch64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEFEC %s
// UNDEFEC:      lld-link: error: undefined symbol: sym
// UNDEFEC-NEXT: >>> referenced by symref-arm64ec.obj:(.data)

// RUN: not lld-link -machine:arm64x -dll -noentry -out:out.dll symref-x86_64.obj sym-aarch64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEFX86 %s
// UNDEFX86:      lld-link: error: undefined symbol: sym
// UNDEFX86-NEXT: >>> referenced by symref-x86_64.obj:(.data)

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err2.dll symref-aarch64.obj sym-x86_64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEF %s

// Check that ARM64X target can have the same symbol names in both native and EC namespaces.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out.dll symref-aarch64.obj sym-aarch64.obj \
// RUN:           symref-arm64ec.obj sym-x86_64.obj

// Check that ARM64X target can reference both native and EC symbols from an archive.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out2.dll symref-aarch64.obj symref-arm64ec.obj sym.lib

// Check that EC object files can reference x86_64 library symbols.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out3.dll symref-arm64ec.obj sym-x86_64.lib
// RUN: lld-link -machine:arm64x -dll -noentry -out:out4.dll symref-arm64ec.obj sym-ll.lib
// RUN: lld-link -machine:arm64x -dll -noentry -out:out5.dll symref-arm64ec.obj sym-imp.lib

// Check that native object files can't reference x86_64 library symbols.

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err3.dll symref-aarch64.obj sym-x86_64.lib \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEF %s

// Check that native object files can reference native library symbols.

// RUN: lld-link -machine:arm64x -dll -noentry -out:out6.dll symref-aarch64.obj sym-aarch64.lib

// Check that EC object files can't reference native ARM64 library symbols.

// RUN: not lld-link -machine:arm64x -dll -noentry -out:err4.dll symref-arm64ec.obj sym-aarch64.lib \
// RUN:              2>&1 | FileCheck --check-prefix=UNDEFEC %s

#--- symref.s
    .data
    .rva sym

    .text
    .globl __icall_helper_arm64ec
__icall_helper_arm64ec:
    ret

#--- sym.s
     .data
     .globl sym
sym:
     .word 0

#--- sym.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc19.33.0"

@sym = dso_local global i32 0, align 4

#--- sym.def
LIBRARY test.dll
EXPORTS
        Func
        sym
