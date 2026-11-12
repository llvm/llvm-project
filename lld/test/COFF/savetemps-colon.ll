; REQUIRES: x86
; RUN: rm -fr %t.dir/savetemps-colon
; RUN: mkdir -p %t.dir/savetemps-colon
; RUN: opt -thinlto-bc -o %t.dir/savetemps-colon/savetemps.obj %s
; RUN: opt -thinlto-bc -o %t.dir/savetemps-colon/thin1.obj %S/Inputs/thinlto.ll

;; Check preopt
; RUN: lld-link /lldsavetemps:preopt /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.obj.*.preopt.bc | count 2

;; Check promote
; RUN: lld-link /lldsavetemps:promote /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.obj.*.promote.bc | count 2

;; Check internalize
; RUN: lld-link /lldsavetemps:internalize /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.obj.*.internalize.bc | count 2

;; Check import
; RUN: lld-link /lldsavetemps:import /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.obj.*.import.bc | count 2

;; Check opt
;; Not supported on Windows due to difficulty with escaping "opt" across platforms.

;; Check precodegen
; RUN: lld-link /lldsavetemps:precodegen /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.obj.*.precodegen.bc | count 2

;; Check combinedindex
; RUN: lld-link /lldsavetemps:combinedindex /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.exe.index.bc | count 1

;; Check prelink
; RUN: lld-link /lldsavetemps:prelink /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.exe.lto.*.obj | count 2

;; Check resolution
; RUN: lld-link /lldsavetemps:resolution /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj
; RUN: ls %t.dir/savetemps-colon/*.resolution.txt | count 1

;; Check error message
; RUN: not lld-link /lldsavetemps:notastage /out:%t.dir/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %t.dir/savetemps-colon/savetemps.obj %t.dir/savetemps-colon/thin1.obj 2>&1 \
; RUN: | FileCheck %s
; CHECK: unknown /lldsavetemps value: notastage

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @g()

define i32 @main() {
  call void @g()
  ret i32 0
}
