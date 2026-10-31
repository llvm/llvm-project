; REQUIRES: x86

; Keep a weak COFF alias that is referenced by a vtable-like table. The alias
; is represented in LLD as a weak-external fallback, but LTO must retain one
; prevailing IR copy so the backend emits that weak-external alias.

; RUN: opt -module-summary -mtriple=x86_64-pc-windows-msvc %s -o %t.obj
; RUN: lld-link /lldsavetemps /out:%t.exe /entry:main %t.obj
; RUN: llvm-nm %t.exe.lto.*.obj | FileCheck %s

; CHECK-DAG: {{[0-9A-Fa-f]+}} {{[TtWw]}} vector_dtor
; CHECK-DAG: {{[0-9A-Fa-f]+}} {{[TtWw]}} scalar_dtor

@vftable = constant [1 x ptr] [ptr @vector_dtor]

define i32 @main() {
  %entry = load ptr, ptr @vftable
  call void %entry()
  ret i32 0
}

@vector_dtor = weak alias void (), ptr @scalar_dtor

define weak void @scalar_dtor() {
  ret void
}
