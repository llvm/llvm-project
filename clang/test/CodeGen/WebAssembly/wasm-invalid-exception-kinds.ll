; RUN: not %clang_cc1 -triple wasm32 -exception-model=arst -S %s 2>&1 | FileCheck -check-prefix=INVALID-VALUE %s

; Make sure invalid values are rejected for -exception-model when the
; input is IR.

; INVALID-VALUE: error: invalid value 'arst' in '-exception-model=arst'

target triple = "wasm32"
