; REQUIRES: x86
; RUN: rm -fr %t.dir/thinlto-archives
; RUN: mkdir -p %t.dir/thinlto-archives %t.dir/thinlto-archives/a %t.dir/thinlto-archives/b
; RUN: opt -thinlto-bc -o %t.dir/thinlto-archives/main.obj %s
; RUN: opt -thinlto-bc -o %t.dir/thinlto-archives/a/bar.obj %S/Inputs/lto-dep.ll
; RUN: opt -thinlto-bc -o %t.dir/thinlto-archives/b/bar.obj %S/Inputs/bar.ll
; RUN: llvm-ar crs %t.dir/thinlto-archives/a.lib %t.dir/thinlto-archives/a/bar.obj
; RUN: llvm-ar crs %t.dir/thinlto-archives/b.lib %t.dir/thinlto-archives/b/bar.obj
; RUN: lld-link -out:%t.dir/thinlto-archives/main.exe -entry:main \
; RUN:     -lldsavetemps -subsystem:console %t.dir/thinlto-archives/main.obj \
; RUN:     %t.dir/thinlto-archives/a.lib %t.dir/thinlto-archives/b.lib
; RUN: FileCheck %s < %t.dir/thinlto-archives/main.exe.resolution.txt

; CHECK: {{/thinlto-archives/main.obj$}}
; CHECK: {{^-r=.*/thinlto-archives/main.obj,main,px$}}
; CHECK: {{/thinlto-archives/a.libbar.obj[0-9]+$}}
; CHECK-NEXT: {{^-r=.*/thinlto-archives/a.libbar.obj[0-9]+,foo,p$}}
; CHECK-NEXT: {{/thinlto-archives/b.libbar.obj[0-9]+$}}
; CHECK-NEXT: {{^-r=.*/thinlto-archives/b.libbar.obj[0-9]+,bar,p$}}

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @bar()
declare void @foo()

define i32 @main() {
  call void @foo()
  call void @bar()
  ret i32 0
}
