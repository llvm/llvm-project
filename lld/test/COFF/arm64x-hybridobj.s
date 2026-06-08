// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym.s -o sym-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym.s -o sym-arm64ec.obj
// RUN: llvm-objcopy --add-section=.obj.arm64ec=sym-arm64ec.obj --set-section-flags=.obj.arm64ec=debug \
// RUN:              sym-arm64.obj sym.obj

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows ref.s -o ref-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows ref.s -o ref-arm64ec.obj
// RUN: llvm-objcopy --add-section=.obj.arm64ec=ref-arm64ec.obj --set-section-flags=.obj.arm64ec=debug \
// RUN:              ref-arm64.obj ref.obj

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj
// RUN: llvm-objcopy --add-section=.obj.arm64ec=loadconfig-arm64ec.obj --set-section-flags=.obj.arm64ec=debug \
// RUN:              loadconfig-arm64.obj loadconfig.obj

// RUN: lld-link -machine:arm64x -dll -noentry -out:out.dll sym.obj loadconfig.obj
// RUN: llvm-readobj --coff-exports out.dll | FileCheck %s

// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-ec.dll sym.obj loadconfig.obj
// RUN: lld-link -machine:arm64 -dll -noentry -out:out-native.dll sym.obj loadconfig.obj

// RUN: llvm-ar cr sym.lib sym.obj
// RUN: lld-link -machine:arm64x -dll -noentry -out:out2.dll ref.obj sym.lib loadconfig.obj
// RUN: llvm-readobj --coff-exports out2.dll | FileCheck %s
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-ec2.dll sym.obj loadconfig.obj
// RUN: lld-link -machine:arm64 -dll -noentry -out:out-native2.dll sym.obj loadconfig.obj

// RUN: llvm-ar cr --thin sym-thin.lib sym.obj
// RUN: lld-link -machine:arm64x -dll -noentry -out:out3.dll ref.obj sym-thin.lib loadconfig.obj
// RUN: llvm-readobj --coff-exports out3.dll | FileCheck %s

// RUN: lld-link -machine:arm64x -dll -noentry -out:out4.dll ref.obj -start-lib sym.obj loadconfig.obj -end-lib
// RUN: llvm-readobj --coff-exports out4.dll | FileCheck %s

// RUN: lld-link -machine:arm64x -dll -noentry -out:out4.dll -wholearchive:sym-thin.lib loadconfig.obj
// RUN: llvm-readobj --coff-exports out4.dll | FileCheck %s

// CHECK:      Format: COFF-ARM64X
// CHECK-NEXT: Arch: aarch64
// CHECK-NEXT: AddressSize: 64bit
// CHECK-NEXT: Export {
// CHECK-NEXT:   Ordinal: 1
// CHECK-NEXT:   Name: sym
// CHECK-NEXT:   RVA: 0x4004
// CHECK-NEXT: }
// CHECK-NEXT: HybridObject {
// CHECK-NEXT:   Format: COFF-ARM64EC
// CHECK-NEXT:   Arch: aarch64
// CHECK-NEXT:   AddressSize: 64bit
// CHECK-NEXT:   Export {
// CHECK-NEXT:     Ordinal: 1
// CHECK-NEXT:     Name: sym
// CHECK-NEXT:     RVA: 0x4000
// CHECK-NEXT:   }
// CHECK-NEXT: }

#--- sym.s
        .section .sym,"dr"
        .globl sym
sym:
        .long 0

        .section .drectve
        .ascii "-export:sym"

#--- ref.s
        .data
        .rva sym
