REQUIRES: aarch64
RUN: split-file %s %t.dir && cd %t.dir

#--- text-func.s
    .text
    .globl func
    .p2align 2, 0x0
func:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    ret

    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows text-func.s -o text-func.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry -out:test.dll text-func.obj 2>&1 | FileCheck -check-prefix=FUNC-NON-COMDAT %s
// FUNC-NON-COMDAT: error: non COMDAT symbol 'func' in hybrid map

#--- offset-func.s
    .section .text,"xr",discard,func
    // Add an instruction before func label to make adding entry thunk offset in the padding impossible.
    mov w0, #2
    .globl func
    .p2align 2, 0x0
func:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    ret

    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows offset-func.s -o offset-func.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry -out:test.dll offset-func.obj 2>&1 | FileCheck -check-prefix=FUNC-NON-COMDAT %s

#--- undef-func.s
    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    ret

    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows undef-func.s -o undef-func.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry -out:test.dll undef-func.obj 2>&1 | FileCheck -check-prefix=UNDEF-FUNC %s
// UNDEF-FUNC: error: undefined symbol: func

#--- undef-thunk.s
    .section .text,"xr",discard,func
    .globl func
    .p2align 2, 0x0
func:
    mov w0, #1
    ret

    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows undef-thunk.s -o undef-thunk.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry -out:test.dll undef-thunk.obj 2>&1 | FileCheck -check-prefix=UNDEF-THUNK %s
// UNDEF-THUNK: error: undefined symbol: thunk

#--- invalid-type.s
    .section .text,"xr",discard,func
    .globl func
    .p2align 2, 0x0
func:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    ret

    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk
    .word 3

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows invalid-type.s -o invalid-type.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:test.dll invalid-type.obj 2>&1 | FileCheck -check-prefix=INVALID-TYPE %s
// INVALID-TYPE: warning: Ignoring unknown EC thunk type 3

#--- invalid-size.s
    .section .hybmp$x,"yi"
    .symidx func
    .symidx thunk

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows invalid-size.s -o invalid-size.obj
// RUN: not lld-link -machine:arm64ec -dll -noentry -out:test.dll invalid-size.obj 2>&1 | FileCheck -check-prefix=INVALID-SIZE %s
// INVALID-SIZE: error: Invalid .hybmp chunk size 8
