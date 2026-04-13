; REQUIRES: x86
; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: llc %t.dir/t1.ll -o %t.dir/t1.obj --filetype=obj
; RUN: llc %t.dir/t2.ll -o %t.dir/t2.obj --filetype=obj
; RUN: lld-link %t.dir/t1.obj %t.dir/t2.obj -entry:main -out:%t.dir/out.exe
; RUN: llvm-readobj --section-headers %t.dir/out.exe | FileCheck %s

; Make sure that the data section contains just one copy of @a, not two.
; CHECK: Name: .data
; CHECK-NEXT: VirtualSize: 0x1000

;--- t1.ll
target triple = "x86_64-pc-windows-msvc"
@a = common global [4096 x i8] zeroinitializer

define i32 @usea() {
  %ref_common = load i32, ptr @a
  ret i32 %ref_common
}

;--- t2.ll
target triple = "x86_64-pc-windows-msvc"
@a = common global [4096 x i8] zeroinitializer

define i32 @useb() {
  %ref_common = load i32, ptr @a
  ret i32 %ref_common
}

declare i32 @usea()

define dso_local i32 @main() local_unnamed_addr {
entry:
  %a = tail call i32 @usea()
  %b = tail call i32 @useb()
  %add = add nsw i32 %a, %b
  ret i32 %add
}
