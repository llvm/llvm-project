// REQUIRES: aarch64, x86
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows symref.s -o symref-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows symref.s -o symref-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym.s -o sym-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym.s -o sym-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows undefref.s -o undefref-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows undefref.s -o undefref-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj

// RUN: rm -f thin.lib
// RUN: llvm-ar rcs --thin thin.lib sym-arm64ec.obj sym-aarch64.obj undefref-arm64ec.obj undefref-aarch64.obj loadconfig-arm64ec.obj loadconfig-aarch64.obj

// Test linking an ARM64EC module against a thin library containing both EC and native symbols.

// RUN: lld-link -machine:arm64ec -dll -noentry -out:test-arm64ec.dll symref-arm64ec.obj thin.lib
// RUN: llvm-readobj --coff-exports test-arm64ec.dll | FileCheck --check-prefix=EXPORTS-ARM64EC %s
// EXPORTS-ARM64EC:      Format: COFF-ARM64EC
// EXPORTS-ARM64EC-NEXT: Arch: aarch64
// EXPORTS-ARM64EC-NEXT: AddressSize: 64bit
// EXPORTS-ARM64EC-NEXT: Export {
// EXPORTS-ARM64EC-NEXT:   Ordinal: 1
// EXPORTS-ARM64EC-NEXT:   Name: sym
// EXPORTS-ARM64EC-NEXT:   RVA:
// EXPORTS-ARM64EC-NEXT: }

// Test linking an ARM64X module referencing both EC and native symbols.

// RUN: lld-link -machine:arm64x -dll -noentry -out:test-arm64x.dll symref-arm64ec.obj symref-aarch64.obj thin.lib
// RUN: llvm-readobj --coff-exports test-arm64x.dll | FileCheck --check-prefix=EXPORTS-ARM64X %s
// EXPORTS-ARM64X:      Format: COFF-ARM64X
// EXPORTS-ARM64X-NEXT: Arch: aarch64
// EXPORTS-ARM64X-NEXT: AddressSize: 64bit
// EXPORTS-ARM64X-NEXT: Export {
// EXPORTS-ARM64X-NEXT:   Ordinal: 1
// EXPORTS-ARM64X-NEXT:   Name: sym
// EXPORTS-ARM64X-NEXT:   RVA:
// EXPORTS-ARM64X-NEXT: }
// EXPORTS-ARM64X-NEXT: HybridObject {
// EXPORTS-ARM64X-NEXT:   Format: COFF-ARM64EC
// EXPORTS-ARM64X-NEXT:   Arch: aarch64
// EXPORTS-ARM64X-NEXT:   AddressSize: 64bit
// EXPORTS-ARM64X-NEXT:   Export {
// EXPORTS-ARM64X-NEXT:     Ordinal: 1
// EXPORTS-ARM64X-NEXT:     Name: sym
// EXPORTS-ARM64X-NEXT:     RVA:
// EXPORTS-ARM64X-NEXT:   }
// EXPORTS-ARM64X-NEXT: }

// Test linking an ARM64X module referencing only EC symbol.

// RUN: lld-link -machine:arm64x -dll -noentry -out:test-arm64x-ecref.dll symref-arm64ec.obj thin.lib
// RUN: llvm-readobj --coff-exports test-arm64x-ecref.dll | FileCheck --check-prefix=EXPORTS-ARM64X2 %s
// EXPORTS-ARM64X2:      Format: COFF-ARM64X
// EXPORTS-ARM64X2-NEXT: Arch: aarch64
// EXPORTS-ARM64X2-NEXT: AddressSize: 64bit
// EXPORTS-ARM64X2-NEXT: HybridObject {
// EXPORTS-ARM64X2-NEXT:   Format: COFF-ARM64EC
// EXPORTS-ARM64X2-NEXT:   Arch: aarch64
// EXPORTS-ARM64X2-NEXT:   AddressSize: 64bit
// EXPORTS-ARM64X2-NEXT:   Export {
// EXPORTS-ARM64X2-NEXT:     Ordinal: 1
// EXPORTS-ARM64X2-NEXT:     Name: sym
// EXPORTS-ARM64X2-NEXT:     RVA:
// EXPORTS-ARM64X2-NEXT:   }
// EXPORTS-ARM64X2-NEXT: }

// Test linking an ARM64X module referencing only native symbol.

// RUN: lld-link -machine:arm64x -dll -noentry -out:test-arm64x-nativeref.dll symref-aarch64.obj thin.lib
// RUN: llvm-readobj --coff-exports test-arm64x-nativeref.dll | FileCheck --check-prefix=EXPORTS-ARM64X3 %s
// EXPORTS-ARM64X3:      Format: COFF-ARM64X
// EXPORTS-ARM64X3-NEXT: Arch: aarch64
// EXPORTS-ARM64X3-NEXT: AddressSize: 64bit
// EXPORTS-ARM64X3-NEXT: Export {
// EXPORTS-ARM64X3-NEXT:   Ordinal: 1
// EXPORTS-ARM64X3-NEXT:   Name: sym
// EXPORTS-ARM64X3-NEXT:   RVA:
// EXPORTS-ARM64X3-NEXT: }
// EXPORTS-ARM64X3-NEXT: HybridObject {
// EXPORTS-ARM64X3-NEXT:   Format: COFF-ARM64EC
// EXPORTS-ARM64X3-NEXT:   Arch: aarch64
// EXPORTS-ARM64X3-NEXT:   AddressSize: 64bit
// EXPORTS-ARM64X3-NEXT: }

#--- symref.s
    .data
    .rva sym

#--- sym.s
     .data
     .globl sym
sym:
     .word 0
     .section .drectve, "yn"
     .ascii " -export:sym,DATA"

#--- undefref.s
    .data
    .globl undefref
undefref:
    .rva undefsym

