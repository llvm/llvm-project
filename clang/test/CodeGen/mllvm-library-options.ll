;; -mllvm options of per-context libraries reach the LLVMContext that compiles.
; RUN: %clang_cc1 -O1 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,SINK
; RUN: %clang_cc1 -O1 -emit-llvm -mllvm -instcombine-code-sinking=0 %s -o - | FileCheck %s --check-prefixes=CHECK,NOSINK

; CHECK-LABEL: define {{.*}} @f(
; NOSINK-NEXT: entry:
; NOSINK-NEXT:  = sdiv
; CHECK:       {{^}}t:
; SINK-NEXT:     = sdiv
define void @f(i1 %c, i32 %a, i32 %b) {
entry:
  %d = sdiv i32 %a, %b
  br i1 %c, label %t, label %e
t:
  call void @use(i32 %d)
  ret void
e:
  ret void
}

declare void @use(i32)
