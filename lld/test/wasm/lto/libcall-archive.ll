; RUN: rm -f %t.a
; RUN: llvm-as -o %t.o %s
; RUN: llvm-as -o %t2.o %S/Inputs/libcall-archive.ll
; RUN: llvm-ar rcs %t.a %t2.o
; RUN: wasm-ld -o %t %t.o %t.a
; RUN: obj2yaml %t | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@llvm.used = appending global [2 x ptr] [ptr @test_acosf, ptr @test___umodti3]

define void @_start(ptr %a, ptr %b) #0 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1024, i1 false)
  ret void
}

; Emit acosf, which currently happens to be the first runtime libcall
; entry.
define float @test_acosf(float %x) {
  %acos = call float @llvm.acos.f32(float %x)
  ret float %acos
}

; Emit __umodti3, which currently happens to be the last runtime
; libcall entry.
define i128 @test___umodti3(i128 %a, i128 %b) {
  %urem = urem i128 %a, %b
  ret i128 %urem
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)

attributes #0 = { "target-features"="-bulk-memory,-bulk-memory-opt" }

; CHECK:       - Type:            CUSTOM
; CHECK-NEXT:    Name:            name
; CHECK-NEXT:    FunctionNames:
; CHECK-NEXT:      - Index:           0
; CHECK-NEXT:        Name:            test_acosf
; CHECK-NEXT:      - Index:           1
; CHECK-NEXT:        Name:            acosf
; CHECK-NEXT:      - Index:           2
; CHECK-NEXT:        Name:            test___umodti3
; CHECK-NEXT:      - Index:           3
; CHECK-NEXT:        Name:            __umodti3
; CHECK-NEXT:      - Index:           4
; CHECK-NEXT:        Name:            _start
