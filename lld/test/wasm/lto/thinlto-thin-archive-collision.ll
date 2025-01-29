; Copied from lld/test/ELF/lto/thinlto-thin-archive-collision.ll

; RUN: rm -fr %t && mkdir %t && cd %t
; RUN: mkdir thinlto-archives thinlto-archives/a thinlto-archives/b
; RUN: opt -thinlto-bc -o thinlto-archives/main.o %s
; RUN: opt -thinlto-bc -o thinlto-archives/a/thin.o %S/Inputs/thin1.ll
; RUN: opt -thinlto-bc -o thinlto-archives/b/thin.o %S/Inputs/thin2.ll
; RUN: llvm-ar qcT thinlto-archives/thin.a thinlto-archives/a/thin.o thinlto-archives/b/thin.o
; RUN: wasm-ld thinlto-archives/main.o thinlto-archives/thin.a -o thinlto-archives/main.exe --save-temps
; RUN: FileCheck %s < thinlto-archives/main.exe.resolution.txt

; CHECK: thinlto-archives/main.o
; CHECK: thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}})
; CHECK-NEXT: -r=thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}}),foo,p
; CHECK: thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}})
; CHECK-NEXT: -r=thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}}),blah,p

target triple = "wasm32-unknown-unknown"

declare i32 @blah(i32 %meh)
declare i32 @foo(i32 %goo)

define i32 @_start() {
  call i32 @foo(i32 0)
  call i32 @blah(i32 0)
  ret i32 0
}
