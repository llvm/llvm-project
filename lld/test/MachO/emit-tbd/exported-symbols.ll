; RUN: rm -rf %t
; RUN: split-file %s %t

; RUN: llvm-as %t/obj.ll -o %t/obj.o

; RUN: echo "_default_exported_symbol_1" > %t/exported-symbol-list.txt

; RUN: %lld -dylib -exported_symbols_list %t/exported-symbol-list.txt -arch arm64 %t/obj.o -o %t/generated.dylib
; RUN: %lld -dylib -exported_symbols_list %t/exported-symbol-list.txt -arch arm64 --emit-tbd-only=%t/lld-generated.tbd %t/obj.o -o %t/generated.dylib

; RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4

; RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

;--- obj.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@default_exported_symbol_1 = global i32 42
@default_exported_symbol_2 = global i32 42
