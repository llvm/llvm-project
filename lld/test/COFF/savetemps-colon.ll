; REQUIRES: x86
; RUN: rm -fr %T/savetemps-colon
; RUN: mkdir %T/savetemps-colon
; RUN: opt -thinlto-bc -o %T/savetemps-colon/savetemps.obj %s
; RUN: opt -thinlto-bc -o %T/savetemps-colon/thin1.obj %S/Inputs/thinlto.ll

;; Check preopt
; RUN: lld-link /lldsavetemps:preopt /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.obj.*.preopt.bc | count 2

;; Check promote
; RUN: lld-link /lldsavetemps:promote /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.obj.*.promote.bc | count 2

;; Check internalize
; RUN: lld-link /lldsavetemps:internalize /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.obj.*.internalize.bc | count 2

;; Check import
; RUN: lld-link /lldsavetemps:import /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.obj.*.import.bc | count 2

;; Check opt
;; Not supported on Windows due to difficulty with escaping "opt" across platforms.

;; Check precodegen
; RUN: lld-link /lldsavetemps:precodegen /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.obj.*.precodegen.bc | count 2

;; Check combinedindex
; RUN: lld-link /lldsavetemps:combinedindex /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.exe.index.bc | count 1

;; Check prelink
; RUN: lld-link /lldsavetemps:prelink /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.exe.lto.*.obj | count 2

;; Check resolution
; RUN: lld-link /lldsavetemps:resolution /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj
; RUN: ls %T/savetemps-colon/*.resolution.txt | count 1

;; Check error message
; RUN: not lld-link /lldsavetemps:notastage /out:%T/savetemps-colon/savetemps.exe /entry:main \
; RUN:     /subsystem:console %T/savetemps-colon/savetemps.obj %T/savetemps-colon/thin1.obj 2>&1 \
; RUN: | FileCheck %s
; CHECK: unknown /lldsavetemps value: notastage

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @g()

define i32 @main() {
  call void @g()
  ret i32 0
}
