; RUN: rm -rf %t
; RUN: split-file %s %t

; RUN: llvm-as %t/bitcode_external.ll -o %t/bitcode_external.o
; RUN: llvm-as %t/bitcode_private_external.ll -o %t/bitcode_private_external.o
; RUN: echo %t/bitcode_external.o > %t/filelist.txt
; RUN: echo %t/bitcode_private_external.o >> %t/filelist.txt
; RUN: echo %t/bitcode_private_external.o > %t/filelist-reversed.txt
; RUN: echo %t/bitcode_external.o >> %t/filelist-reversed.txt

; RUN: %lld -dylib -lSystem -arch arm64 -filelist %t/filelist.txt -o %t/generated.dylib
; RUN: %lld -dylib -lSystem -arch arm64 --emit-tbd-only=%t/lld-generated.tbd -filelist %t/filelist.txt -o %t/generated.dylib
; RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4
; RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

; RUN: %lld -dylib -lSystem -arch arm64 -filelist %t/filelist-reversed.txt -o %t/generated-reversed.dylib
; RUN: %lld -dylib -lSystem -arch arm64 --emit-tbd-only=%t/lld-generated-reversed.tbd -filelist %t/filelist-reversed.txt -o %t/generated-reversed.dylib
; RUN: llvm-readtapi -stubify %t/generated-reversed.dylib -o %t/tapi-generated-reversed.tbd --filetype=tbd-v4
; RUN: llvm-readtapi -compare %t/lld-generated-reversed.tbd %t/tapi-generated-reversed.tbd

;--- bitcode_external.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@foo = weak_odr global i32 0, align 4

;--- bitcode_private_external.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@foo = weak_odr hidden global i32 0, align 4
