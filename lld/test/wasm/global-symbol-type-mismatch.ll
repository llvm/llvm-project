; RUN: split-file %s %t
; RUN: llc -mtriple=wasm32-unknown-unknown -filetype=obj %t/use-global.ll -o %t/use-global.o
; RUN: llc -mtriple=wasm32-unknown-unknown -filetype=obj %t/define-global.ll -o %t/define-global.o
; RUN: llc -mtriple=wasm32-unknown-unknown -filetype=obj %t/define-data.ll -o %t/define-data.o
; RUN: wasm-ld --no-entry --export=read_global %t/use-global.o %t/define-global.o -o %t/matching.wasm
; RUN: not wasm-ld --no-entry --export=read_global %t/use-global.o %t/define-data.o -o %t/mismatch.wasm 2>&1 | FileCheck %s --check-prefix=GLOBAL-FIRST
; RUN: not wasm-ld --no-entry --export=read_global %t/define-data.o %t/use-global.o -o %t/mismatch.wasm 2>&1 | FileCheck %s --check-prefix=DATA-FIRST

; GLOBAL-FIRST: error: symbol type mismatch: storage_global
; GLOBAL-FIRST-NEXT: >>> defined as WASM_SYMBOL_TYPE_GLOBAL in {{.*}}use-global.o
; GLOBAL-FIRST-NEXT: >>> defined as WASM_SYMBOL_TYPE_DATA in {{.*}}define-data.o

; DATA-FIRST: error: symbol type mismatch: storage_global
; DATA-FIRST-NEXT: >>> defined as WASM_SYMBOL_TYPE_DATA in {{.*}}define-data.o
; DATA-FIRST-NEXT: >>> defined as WASM_SYMBOL_TYPE_GLOBAL in {{.*}}use-global.o

;--- use-global.ll
@storage_global = external addrspace(1) global i32

define i32 @read_global() {
  %value = load i32, ptr addrspace(1) @storage_global, align 4
  ret i32 %value
}

;--- define-global.ll
@storage_global = addrspace(1) global i32 0, align 4

;--- define-data.ll
@storage_global = global i32 0, align 4