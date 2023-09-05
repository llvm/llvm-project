; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: llc -filetype=obj %t/foo1.ll -o %t/foo1.o
; RUN: llc -filetype=obj %t/foo2.ll -o %t/foo2.o
; RUN: llvm-ar rcs %t/libfoo2.a %t/foo2.o
; RUN: llc -filetype=obj %t/foo3.ll -o %t/foo3.o
; RUN: llvm-ar rcs %t/libfoo3.a %t/foo3.o

; RUN: llc -filetype=obj %t/zoo2.ll -o %t/zoo2.o
; RUN: llvm-ar rcs %t/libzoo2.a %t/zoo2.o
; RUN: llc -filetype=obj %t/zoo3.ll -o %t/zoo3.o
; RUN: llvm-ar rcs %t/libzoo3.a %t/zoo3.o

; RUN: llc -filetype=obj %t/bar1.ll -o %t/bar1.o
; RUN: llc -filetype=obj %t/bar2.ll -o %t/bar2.o
; RUN: llvm-ar rcs %t/libbar2.a %t/bar2.o
; RUN: llc -filetype=obj %t/bar3.ll -o %t/bar3.o
; RUN: llvm-ar rcs %t/libbar3.a %t/bar3.o

; RUN: %lld -dylib -lSystem -L%t %t/foo1.o %t/bar1.o -o %t/order.out
; RUN: llvm-objdump --no-leading-addr --no-show-raw-insn -d %t/order.out | FileCheck %s

; We want to process input object files first
; before any lc-linker options are actually resolved.
; The lc-linker options are recursively processed.

; The following shows a chain of auto linker options,
; starting with foo1.o and bar1.o:
;
; foo1.o -> libfoo2.a(foo2.o) -> libfoo3.a(foo3.o)
;       \
;        -> libzoo2.a(zoo2.o) -> libzoo3.a(zoo3.o)
; bar1.o -> libbar2.a(bar2.o) -> libbar3.a(bar3.o)

; CHECK: <_foo1>:
; CHECK: <_bar1>:
; CHECK: <_foo2>:
; CHECK: <_zoo2>:
; CHECK: <_bar2>:
; CHECK: <_foo3>:
; CHECK: <_zoo3>:
; CHECK: <_bar3>:

;--- foo1.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lfoo2"}
!1 = !{!"-lzoo2"}
!llvm.linker.options = !{!0, !1}

define i32 @foo1() {
  %call = call i32 @foo2()
  %call2 = call i32 @zoo2()
  %add = add nsw i32 %call, %call2
  ret i32 %add
}

declare i32 @foo2()
declare i32 @zoo2()

;--- foo2.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lfoo3"}
!llvm.linker.options = !{!0}

define i32 @foo2() {
  %call = call i32 @foo3()
  %add = add nsw i32 %call, 2
  ret i32 %add
}

declare i32 @foo3()

;--- foo3.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @foo3() {
  ret i32 3
}

;--- zoo2.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lzoo3"}
!llvm.linker.options = !{!0}

define i32 @zoo2() {
  %call = call i32 @zoo3()
  %add = add nsw i32 %call, 2
  ret i32 %add
}

declare i32 @zoo3()

;--- zoo3.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @zoo3() {
  ret i32 30
}

;--- bar1.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lbar2"}
!llvm.linker.options = !{!0}

define i32 @bar1() {
  %call = call i32 @bar2()
  %add = add nsw i32 %call, 10
  ret i32 %add
}

declare i32 @bar2()

;--- bar2.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

!0 = !{!"-lbar3"}
!llvm.linker.options = !{!0}

define i32 @bar2() {
  %call = call i32 @bar3()
  %add = add nsw i32 %call, 200
  ret i32 %add
}

declare i32 @bar3()

;--- bar3.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

define i32 @bar3() {
  ret i32 300
}
