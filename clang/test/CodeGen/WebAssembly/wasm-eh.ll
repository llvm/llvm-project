; REQUIRES: webassembly-registered-target
; RUN: %clang %s -target wasm32-unknown-unknown -fwasm-exceptions -c -S -o - | FileCheck %s

; This tests whether clang driver can take -fwasm-exceptions and compile bitcode
; files using Wasm EH.

; CHECK-LABEL: test
; CHECK: try
; CHECK:   call foo
; CHECK: catch __cpp_exception
; CHECK: end
define void @test() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) #2 [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch.start
  ret void
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
declare ptr @llvm.wasm.get.exception(token)
declare i32 @llvm.wasm.get.ehselector(token)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

