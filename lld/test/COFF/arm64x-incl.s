// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows sym-arm64ec.s -o sym-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows sym-aarch64.s -o sym-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows drectve.s -o drectve-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows drectve.s -o drectve-aarch64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64.s -o loadconfig-arm64.obj
// RUN: llvm-lib -machine:arm64x -out:sym.lib sym-arm64ec.obj sym-aarch64.obj

// Check that the command-line -include argument ensures the EC symbol is included.

// RUN: lld-link -machine:arm64x -out:out-arg.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj sym.lib -include:sym
// RUN: llvm-readobj --hex-dump=.test out-arg.dll | FileCheck --check-prefix=EC %s
// EC: 0x180004000 02000000                            ....

// Check that the native .drectve -include argument ensures the native symbol is included.

// RUN: lld-link -machine:arm64x -out:out-native.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj sym.lib drectve-aarch64.obj
// RUN: llvm-readobj --hex-dump=.test out-native.dll | FileCheck --check-prefix=NATIVE %s
// NATIVE: 0x180004000 01000000                            ....

// Check that the EC .drectve -include argument ensures the EC symbol is included.

// RUN: lld-link -machine:arm64x -out:out-ec.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj sym.lib drectve-arm64ec.obj
// RUN: llvm-readobj --hex-dump=.test out-ec.dll | FileCheck --check-prefix=EC %s

// Check that both native and EC .drectve -include arguments ensure both symbols are included.

// RUN: lld-link -machine:arm64x -out:out-arg-native.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj sym.lib \
// RUN:          -include:sym drectve-aarch64.obj
// RUN: llvm-readobj --hex-dump=.test out-arg-native.dll | FileCheck --check-prefix=BOTH %s
// BOTH: 0x180004000 02000000 01000000                   ........

// RUN: lld-link -machine:arm64x -out:out-both.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj sym.lib \
// RUN:          drectve-arm64ec.obj drectve-aarch64.obj
// RUN: llvm-readobj --hex-dump=.test out-both.dll | FileCheck --check-prefix=BOTH %s

// Check that including a missing symbol results in an error.

// RUN: not lld-link -machine:arm64x -out:err.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj -include:sym sym-aarch64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=ERR %s
// ERR: lld-link: error: <root>: undefined symbol: sym

// RUN: not lld-link -machine:arm64x -out:err.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj drectve-arm64ec.obj sym-aarch64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=ERR %s

// RUN: not lld-link -machine:arm64x -out:err.dll -dll -noentry loadconfig-arm64.obj loadconfig-arm64ec.obj drectve-aarch64.obj sym-arm64ec.obj \
// RUN:              2>&1 | FileCheck --check-prefix=ERR %s

#--- sym-aarch64.s
        .section ".test","dr"
        .globl sym
sym:
        .word 1

#--- sym-arm64ec.s
        .section ".test","dr"
        .globl sym
sym:
        .word 2

#--- drectve.s
        .section .drectve, "yn"
        .ascii " -include:sym"
