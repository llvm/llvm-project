# RUN: rm -rf %t; split-file %s %t
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %t/f1.ll -o %t/f1.o -mattr=+branch-hinting
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %t/f2.ll -o %t/f2.o -mattr=+branch-hinting
; RUN: wasm-ld --export-all -o %t.wasm %t/f2.o %t/f1.o
; RUN: obj2yaml %t.wasm | FileCheck --check-prefixes=CHECK %s

; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %t/f1.ll -o %t/f1.o -mattr=-branch-hinting
; RUN: llc -mcpu=mvp -mtriple=wasm32-unknown-unknown -filetype=obj %t/f2.ll -o %t/f2.o -mattr=-branch-hinting
; RUN: wasm-ld --export-all -o %t.wasm %t/f2.o %t/f1.o
; RUN: obj2yaml %t.wasm | FileCheck --check-prefixes=NCHECK %s

; CHECK:          - Type:            CUSTOM
; CHECK:            Name:            metadata.code.branch_hint
; CHECK-NEXT:       Entries:
; CHECK-NEXT:         - FuncIdx:         1
; CHECK-NEXT:           Hints:
; CHECK-NEXT:             - Offset:          7
; CHECK-NEXT:               Size:            1
; CHECK-NEXT:               Data:            UNLIKELY
; CHECK-NEXT:             - Offset:          14
; CHECK-NEXT:               Size:            1
; CHECK-NEXT:               Data:            LIKELY
; CHECK-NEXT:         - FuncIdx:         2
; CHECK-NEXT:           Hints:
; CHECK-NEXT:             - Offset:          5
; CHECK-NEXT:               Size:            1
; CHECK-NEXT:               Data:            LIKELY
; CHECK-NEXT:         - FuncIdx:         3
; CHECK-NEXT:           Hints:
; CHECK-NEXT:             - Offset:          5
; CHECK-NEXT:               Size:            1
; CHECK-NEXT:               Data:            UNLIKELY
; CHECK-NEXT:         - FuncIdx:         4
; CHECK-NEXT:           Hints:
; CHECK-NEXT:             - Offset:          5
; CHECK-NEXT:               Size:            1
; CHECK-NEXT:               Data:            LIKELY

; CHECK:         - Type:            CUSTOM
; CHECK-NEXT:      Name:            name
; CHECK-NEXT:      FunctionNames:
; CHECK-NEXT:        - Index:           0
; CHECK-NEXT:          Name:            __wasm_call_ctors
; CHECK-NEXT:        - Index:           1
; CHECK-NEXT:          Name:            test0
; CHECK-NEXT:        - Index:           2
; CHECK-NEXT:          Name:            test1
; CHECK-NEXT:        - Index:           3
; CHECK-NEXT:          Name:            _start
; CHECK-NEXT:        - Index:           4
; CHECK-NEXT:          Name:            test_func1

; CHECK:        - Type:            CUSTOM
; CHECK:          Name:            target_features
; CHECK-NEXT:     Features:
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            branch-hinting


; NCHECK-NOT:         Name:            metadata.code.branch_hint
; NCHECK-NOT:         Name:            branch-hinting

#--- f1.ll
define i32 @_start(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !0
if.then:
  ret i32 1
if.else:
  ret i32 2
}

define i32 @test_func1(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !1
if.then:
  ret i32 1
if.else:
  ret i32 2
}

!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 2000}

#--- f2.ll
target triple = "wasm32-unknown-unknown"

define i32 @test0(i32 %a) {
entry:
  %cmp0 = icmp eq i32 %a, 0
  br i1 %cmp0, label %if.then, label %ret1, !prof !0
if.then:
  %cmp1 = icmp eq i32 %a, 1
  br i1 %cmp1, label %ret1, label %ret2, !prof !1
ret1:
  ret i32 2
ret2:
  ret i32 1
}

define i32 @test1(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else, !prof !1
if.then:
  ret i32 1
if.else:
  ret i32 2
}

; the resulting branch hint is actually reversed, since llvm-br is turned into br_unless, inverting branch probs
!0 = !{!"branch_weights", i32 2000, i32 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
