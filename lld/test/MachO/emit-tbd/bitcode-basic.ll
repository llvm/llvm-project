; RUN: rm -rf %t
; RUN: split-file %s %t

; RUN: llvm-as %t/obj.ll -o %t/obj.o

; RUN: %lld -dylib -arch arm64 %t/obj.o -o %t/generated.dylib
; RUN: %lld -dylib -arch arm64 --emit-tbd-only=%t/lld-generated.tbd %t/obj.o -o %t/generated.dylib

; RUN: llvm-readtapi -stubify %t/generated.dylib -o %t/tapi-generated.tbd --filetype=tbd-v4

; RUN: llvm-readtapi -compare %t/lld-generated.tbd %t/tapi-generated.tbd

;--- obj.ll

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-ios-simulator15.1.0"

@external_linkage_default_visibility_gv = global i32 42
define i32 @external_linkage_default_visibility_func() {
  ret i32 42
}

@internal_linkage_default_visibility_gv = internal global i32 42
define internal i32 @internal_linkage_default_visibility_func() {
  ret i32 42
}

@private_linkage_default_visibility_gv = private global i32 42
define private i32 @private_linkage_default_visibility_func() {
  ret i32 42
}

@weak_linkage_default_visibility_gv = weak global i32 42
define weak i32 @weak_linkage_default_visibility_func() {
  ret i32 42
}

@linkonce_linkage_default_visibility_gv = linkonce global i32 42
define linkonce i32 @linkonce_linkage_default_visibility_func() {
  ret i32 42
}

@linkonce_odr_linkage_default_visibility_gv = linkonce_odr global i32 42
define linkonce_odr i32 @linkonce_odr_linkage_default_visibility_func() {
  ret i32 42
}

@weak_odr_linkage_default_visibility_gv = weak_odr global i32 42
define weak_odr i32 @weak_odr_linkage_default_visibility_func() {
  ret i32 42
}

@common_linkage_default_visibility_gv = common global i32 0

@external_linkage_hidden_visibility_gv = hidden global i32 42
define hidden i32 @external_linkage_hidden_visibility_func() {
  ret i32 42
}

@weak_linkage_hidden_visibility_gv = weak hidden global i32 42
define weak hidden i32 @weak_linkage_hidden_visibility_func() {
  ret i32 42
}

@linkonce_linkage_hidden_visibility_gv = linkonce hidden global i32 42
define linkonce hidden i32 @linkonce_linkage_hidden_visibility_func() {
  ret i32 42
}

@linkonce_odr_linkage_hidden_visibility_gv = linkonce_odr hidden global i32 42
define linkonce_odr hidden i32 @linkonce_odr_linkage_hidden_visibility_func() {
  ret i32 42
}

@weak_odr_linkage_hidden_visibility_gv = weak_odr hidden global i32 42
define weak_odr hidden i32 @weak_odr_linkage_hidden_visibility_func() {
  ret i32 42
}

@common_linkage_hidden_visibility_gv = common hidden global i32 0

@external_linkage_default_visibility_gv_declaration = external global i32

declare i32 @external_linkage_default_visibility_func_declaration()
