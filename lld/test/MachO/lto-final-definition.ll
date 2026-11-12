; REQUIRES: x86
; RUN: rm -rf %t; mkdir %t
; RUN: llvm-as %s -o %t/test.o
; RUN: %lld -lSystem -dylib %t/test.o -o %t/test -save-temps
; RUN: llvm-dis %t/test.0.2.internalize.bc -o - | FileCheck %s
; RUN: %lld -lSystem -dylib %t/test.o -o %t/flat-namespace.dylib -save-temps \
; RUN:   -flat_namespace
; RUN: llvm-dis %t/flat-namespace.dylib.0.2.internalize.bc -o - | FileCheck %s \
; RUN:   --check-prefix=NO-DSO-LOCAL

;; f() is never dso_local since it is a weak external.
; CHECK:        define weak_odr void @f()
; CHECK:        define dso_local void @main()

; NO-DSO-LOCAL: define weak_odr void @f()
; NO-DSO-LOCAL: define void @main()

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define weak_odr void @f() {
  ret void
}

define void @main() {
  ret void
}
