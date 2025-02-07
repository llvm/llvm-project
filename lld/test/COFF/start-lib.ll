; REQUIRES: x86

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir

; RUN: llc -filetype=obj %t.dir/main.ll -o %t.obj
; RUN: llc -filetype=obj %t.dir/start-lib1.ll -o %t1.obj
; RUN: llc -filetype=obj %t.dir/start-lib2.ll -o %t2.obj
; RUN: llc -filetype=obj %t.dir/eager.ll -o %t-eager.obj
; RUN: opt -thinlto-bc %t.dir/main.ll -o %t.bc
; RUN: opt -thinlto-bc %t.dir/start-lib1.ll -o %t1.bc
; RUN: opt -thinlto-bc %t.dir/start-lib2.ll -o %t2.bc
; RUN: opt -thinlto-bc %t.dir/eager.ll -o %t-eager.bc
;
; RUN: lld-link -out:%t1.exe -entry:main -opt:noref -lldmap:%t1.map \
; RUN:     %t.obj %t1.obj %t2.obj %t-eager.obj
; RUN: FileCheck --check-prefix=TEST1 %s < %t1.map
; RUN: lld-link -out:%t1.exe -entry:main -opt:noref -lldmap:%t1.thinlto.map \
; RUN:     %t.bc %t1.bc %t2.bc %t-eager.bc
; RUN: FileCheck --check-prefix=TEST1 %s < %t1.thinlto.map
; TEST1: foo
; TEST1: bar
;
; RUN: lld-link -out:%t2.exe -entry:main -opt:noref -lldmap:%t2.map \
; RUN:     %t.obj -start-lib %t1.obj %t-eager.obj -end-lib %t2.obj
; RUN: FileCheck --check-prefix=TEST2 %s < %t2.map
; RUN: lld-link -out:%t2.exe -entry:main -opt:noref -lldmap:%t2.thinlto.map \
; RUN:     %t.bc -start-lib %t1.bc %t-eager.bc -end-lib %t2.bc
; RUN: FileCheck --check-prefix=TEST2 %s < %t2.thinlto.map
; TEST2:     Address Size Align Out In Symbol
; TEST2-NOT:                           {{ }}foo{{$}}
; TEST2:                               {{ }}bar{{$}}
; TEST2-NOT:                           {{ }}foo{{$}}
;
; RUN: lld-link -out:%t3.exe -entry:main -opt:noref -lldmap:%t3.map \
; RUN:     %t.obj -start-lib %t1.obj %t2.obj %t-eager.obj
; RUN: FileCheck --check-prefix=TEST3 %s < %t3.map
; RUN: lld-link -out:%t3.exe -entry:main -opt:noref -lldmap:%t3.thinlto.map \
; RUN:     %t.bc -start-lib %t1.bc %t2.bc %t-eager.bc
; RUN: FileCheck --check-prefix=TEST3 %s < %t3.thinlto.map
; TEST3:     Address Size Align Out In Symbol
; TEST3-NOT: {{ }}foo{{$}}
; TEST3-NOT: {{ }}bar{{$}}


#--- main.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @eager()

define void @main() {
  call void @eager()
  ret void
}


#--- start-lib1.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @bar()

define i32 @foo() {
  %1 = call i32 () @bar()
  %2 = add i32 %1, 1
  ret i32 %2
}

!llvm.linker.options = !{!0}
!0 = !{!"/INCLUDE:foo"}


#--- start-lib2.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @bar() {
  ret i32 1
}

!llvm.linker.options = !{!0}
!0 = !{!"/INCLUDE:bar"}

#--- eager.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @eager() {
  ret void
}

define i32 @ogre() {
  ret i32 1
}


; Check that lazy object files trigger loads correctly.
; If the links succeed, that's enough, no additional tests needed.

; RUN: llc -filetype=obj %t.dir/main2.ll -o %t-main2.obj
; RUN: llc -filetype=obj %t.dir/foo.ll -o %t-foo.obj
; RUN: llc -filetype=obj %t.dir/bar.ll -o %t-bar.obj
; RUN: llc -filetype=obj %t.dir/baz.ll -o %t-baz.obj
; RUN: opt -thinlto-bc %t.dir/main2.ll -o %t-main2.bc
; RUN: opt -thinlto-bc %t.dir/foo.ll -o %t-foo.bc
; RUN: opt -thinlto-bc %t.dir/bar.ll -o %t-bar.bc
; RUN: opt -thinlto-bc %t.dir/baz.ll -o %t-baz.bc

; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     %t-main2.obj %t-foo.obj %t-bar.obj %t-baz.obj
; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     %t-main2.obj /start-lib %t-foo.obj %t-bar.obj %t-baz.obj /end-lib
; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     /start-lib %t-foo.obj %t-bar.obj %t-baz.obj /end-lib %t-main2.obj

; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     %t-main2.bc %t-foo.bc %t-bar.bc %t-baz.bc
; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     %t-main2.bc /start-lib %t-foo.bc %t-bar.bc %t-baz.bc /end-lib
; RUN: lld-link -out:%t2.exe -entry:main \
; RUN:     /start-lib %t-foo.bc %t-bar.bc %t-baz.bc /end-lib %t-main2.bc

#--- main2.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @bar()

define void @main() {
  call void () @bar()
  ret void
}


#--- foo.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @foo() {
  ret void
}


#--- bar.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; One undefined symbol from the lazy obj file before it,
; one from the one after it.
declare void @foo()
declare void @baz()

define void @bar() {
  call void () @foo()
  call void () @baz()
  ret void
}


#--- baz.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @baz() {
  ret void
}


; Check cycles between symbols in two /start-lib files.
; If the links succeed and does not emit duplicate symbol diagnostics,
; that's enough.

; RUN: llc -filetype=obj %t.dir/main3.ll -o %t-main3.obj
; RUN: llc -filetype=obj %t.dir/cycle1.ll -o %t-cycle1.obj
; RUN: llc -filetype=obj %t.dir/cycle2.ll -o %t-cycle2.obj
; RUN: opt -thinlto-bc %t.dir/main3.ll -o %t-main3.bc
; RUN: opt -thinlto-bc %t.dir/cycle1.ll -o %t-cycle1.bc
; RUN: opt -thinlto-bc %t.dir/cycle2.ll -o %t-cycle2.bc

; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     %t-main3.obj %t-cycle1.obj %t-cycle2.obj
; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     %t-main3.obj /start-lib %t-cycle1.obj %t-cycle2.obj /end-lib
; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     /start-lib %t-cycle1.obj %t-cycle2.obj /end-lib %t-main3.obj

; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     %t-main3.bc %t-cycle1.bc %t-cycle2.bc
; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     %t-main3.bc /start-lib %t-cycle1.bc %t-cycle2.bc /end-lib
; RUN: lld-link -out:%t3.exe -entry:main \
; RUN:     /start-lib %t-cycle1.bc %t-cycle2.bc /end-lib %t-main3.bc

#--- main3.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @foo1()

define void @main() {
  call void () @foo1()
  ret void
}

#--- cycle1.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @bar()

define void @foo1() {
  ; cycle1.ll pulls in cycle2.ll for bar(), and cycle2.ll then pulls in
  ; cycle1.ll again for foo2().
  call void () @bar()
  ret void
}

define void @foo2() {
  ret void
}


#--- cycle2.ll

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @foo2()

define void @bar() {
  call void () @foo2()
  ret void
}
