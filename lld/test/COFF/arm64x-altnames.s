// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=aarch64-windows test.s -o test-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test.s -o test-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows drectve.s -o drectve-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows drectve.s -o drectve-arm64ec.obj

// Check that the -alternatename command-line argument applies only to the EC namespace.

// RUN: not lld-link -out:out.dll -machine:arm64x -dll -noentry test-arm64.obj test-arm64ec.obj -alternatename:sym=altsym \
// RUN:              2>&1 | FileCheck --check-prefix=ERR-NATIVE %s

// ERR-NATIVE-NOT:  test-arm64ec.obj
// ERR-NATIVE:      lld-link: error: undefined symbol: sym (native symbol)
// ERR-NATIVE-NEXT: >>> referenced by test-arm64.obj:(.test)
// ERR-NATIVE-NOT:  test-arm64ec.obj

// Check that the -alternatename .drectve directive applies only to the namespace in which it is defined.

// RUN: not lld-link -out:out.dll -machine:arm64x -dll -noentry test-arm64.obj test-arm64ec.obj drectve-arm64ec.obj \
// RUN:              2>&1 | FileCheck --check-prefix=ERR-NATIVE %s

// RUN: not lld-link -out:out.dll -machine:arm64x -dll -noentry test-arm64.obj test-arm64ec.obj drectve-arm64.obj \
// RUN:              2>&1 | FileCheck --check-prefix=ERR-EC %s

// ERR-EC-NOT:  test-arm64.obj
// ERR-EC:      lld-link: error: undefined symbol: sym (EC symbol)
// ERR-EC-NEXT: >>> referenced by test-arm64ec.obj:(.test)
// ERR-EC-NOT:  test-arm64.obj

// RUN: lld-link -out:out.dll -machine:arm64x -dll -noentry test-arm64.obj test-arm64ec.obj drectve-arm64.obj drectve-arm64ec.obj

#--- test.s
        .section .test,"dr"
        .rva sym
        .data
        .globl altsym
altsym:
        .word 0

#--- drectve.s
        .section .drectve
        .ascii "-alternatename:sym=altsym"
