; Test to ensure -fdiagnostics-show-hotness and -fsave-optimization-record
; work when invoking the ThinLTO backend path.
; REQUIRES: x86-registered-target

; RUN: opt -module-summary -o %t.o %s
; RUN: llvm-lto -thinlto -o %t %t.o

; First try YAML pass remarks file
; RUN: rm -f %t2.opt.yaml
; RUN: %clang -target x86_64-scei-ps4 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -fsave-optimization-record -fdiagnostics-show-hotness -o %t2.o -c
; RUN: cat %t2.opt.yaml | FileCheck %s -check-prefix=YAML

; YAML: --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Hotness:         300
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - Callee:          tinkywinky
; YAML-NEXT:   - String:          ''' inlined into '''
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '0'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

; Next try with pass remarks to stderr
; RUN: %clang -target x86_64-scei-ps4 -O2 -x ir %t.o -fthinlto-index=%t.thinlto.bc -Rpass=inline -fdiagnostics-show-hotness -o %t2.o -c 2>&1 | FileCheck %s

; CHECK: 'tinkywinky' inlined into 'main' with (cost=0, threshold=337) (hotness: 300)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

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
