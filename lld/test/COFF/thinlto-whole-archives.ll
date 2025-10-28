; REQUIRES: x86
; RUN: rm -fr %t.dir/thinlto-whole-archives
; RUN: mkdir -p %t.dir/thinlto-whole-archives %t.dir/thinlto-whole-archives/a %t.dir/thinlto-whole-archives/b
; RUN: opt -thinlto-bc -o %t.dir/thinlto-whole-archives/main.obj %s
; RUN: opt -thinlto-bc -o %t.dir/thinlto-whole-archives/a/bar.obj %S/Inputs/lto-dep.ll
; RUN: opt -thinlto-bc -o %t.dir/thinlto-whole-archives/b/bar.obj %S/Inputs/bar.ll
; RUN: llvm-ar crs %t.dir/thinlto-whole-archives/a.lib %t.dir/thinlto-whole-archives/a/bar.obj %t.dir/thinlto-whole-archives/b/bar.obj
; RUN: lld-link -out:%t.dir/thinlto-whole-archives/main.exe -entry:main \
; RUN:     -wholearchive -lldsavetemps -subsystem:console %t.dir/thinlto-whole-archives/main.obj \
; RUN:     %t.dir/thinlto-whole-archives/a.lib
; RUN: FileCheck %s < %t.dir/thinlto-whole-archives/main.exe.resolution.txt

; CHECK: {{[/\\]thinlto-whole-archives[/\\]main.obj$}}
; CHECK: {{^-r=.*[/\\]thinlto-whole-archives[/\\]main.obj,main,px$}}
; CHECK: {{[/\\]thinlto-whole-archives[/\\]a.libbar.obj[0-9]+$}}
; CHECK-NEXT: {{^-r=.*[/\\]thinlto-whole-archives[/\\]a.libbar.obj[0-9]+,foo,p$}}
; CHECK-NEXT: {{[/\\]thinlto-whole-archives[/\\]a.libbar.obj[0-9]+$}}
; CHECK-NEXT: {{^-r=.*[/\\]thinlto-whole-archives[/\\]a.libbar.obj[0-9]+,bar,p$}}

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @bar()
declare void @foo()

define i32 @main() {
  call void @foo()
  call void @bar()
  ret i32 0
}
