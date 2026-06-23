; REQUIRES: x86
; RUN: llvm-as %s -o %t.obj

; RUN: rm -f %t.yaml %t.pass.yaml %t.hot.yaml %t.t300.yaml %t.t301.yaml
; RUN: lld-link -opt-remarks-filename %t.yaml %t.obj -entry:main -nodefaultlib \
; RUN:   -out:%t.exe -force:unresolved
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML

; RUN: lld-link -opt-remarks-filename %t.pass.yaml -opt-remarks-passes inline \
; RUN:   %t.obj -entry:main -nodefaultlib -out:%t.exe -force:unresolved
; RUN: cat %t.pass.yaml | FileCheck %s -check-prefix=YAML-PASSES

; RUN: lld-link -opt-remarks-with-hotness -opt-remarks-filename %t.hot.yaml \
; RUN:   %t.obj -entry:main -nodefaultlib -out:%t.exe -force:unresolved
; RUN: cat %t.hot.yaml | FileCheck %s -check-prefix=YAML-HOT

; RUN: lld-link -opt-remarks-with-hotness \
; RUN:   -opt-remarks-hotness-threshold:300 \
; RUN:   -opt-remarks-filename %t.t300.yaml %t.obj -entry:main -nodefaultlib \
; RUN:   -out:%t.exe -force:unresolved
; RUN: FileCheck %s -check-prefix=YAML-HOT < %t.t300.yaml

; RUN: lld-link -opt-remarks-with-hotness \
; RUN:   -opt-remarks-hotness-threshold:301 \
; RUN:   -opt-remarks-filename %t.t301.yaml %t.obj -entry:main -nodefaultlib \
; RUN:   -out:%t.exe -force:unresolved
; RUN: count 0 < %t.t301.yaml

; RUN: lld-link -opt-remarks-filename %t.yaml -opt-remarks-format yaml \
; RUN:   %t.obj -entry:main -nodefaultlib -out:%t.exe -force:unresolved
; RUN: FileCheck %s -check-prefix=YAML < %t.yaml

; YAML:      --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - Callee:          tinkywinky
; YAML-NEXT:   - String:          ''' inlined into '''
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

; YAML-HOT:      --- !Passed
; YAML-HOT-NEXT: Pass:            inline
; YAML-HOT-NEXT: Name:            Inlined
; YAML-HOT-NEXT: Function:        main
; YAML-HOT-NEXT: Hotness:         300
; YAML-HOT-NEXT: Args:
; YAML-HOT-NEXT:   - String:          ''''
; YAML-HOT-NEXT:   - Callee:          tinkywinky
; YAML-HOT-NEXT:   - String:          ''' inlined into '''
; YAML-HOT-NEXT:   - Caller:          main
; YAML-HOT-NEXT:   - String:          ''''
; YAML-HOT-NEXT:   - String:          ' with '
; YAML-HOT-NEXT:   - String:          '(cost='
; YAML-HOT-NEXT:   - Cost:
; YAML-HOT-NEXT:   - String:          ', threshold='
; YAML-HOT-NEXT:   - Threshold:
; YAML-HOT-NEXT:   - String:          ')'
; YAML-HOT-NEXT: ...

; YAML-PASSES: Pass:            inline

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

declare i32 @patatino()

define i32 @tinkywinky() {
  %a = call i32 @patatino()
  ret i32 %a
}

define i32 @main() !prof !0 {
  %i = call i32 @tinkywinky()
  ret i32 %i
}

!0 = !{!"function_entry_count", i64 300}
