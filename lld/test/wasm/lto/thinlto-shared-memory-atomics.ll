; RUN: opt -module-summary %s -o %t.main.o
; RUN: opt -module-summary %p/Inputs/thinlto-shared-memory-atomics-empty.ll -o %t.empty.o
; RUN: wasm-ld --shared-memory %t.main.o %t.empty.o -o %t.wasm
; RUN: llvm-nm %t.wasm | FileCheck %s

; This test ensures that when an unused function with target-features="+atomics,+bulk-memory"
; is stripped during ThinLTO, the resulting empty module does not fall back to generic TargetMachine
; features and incorrectly omit the atomics/bulk-memory features from the generated custom sections.
;
; Since --shared-memory is passed to wasm-ld, `+atomics` and `+bulk-memory` are propagated
; to the LTO Config via MAttrs, bypassing the functional absence of attributes.

; CHECK-NOT: error:
; CHECK: _start


target triple = "wasm32-unknown-emscripten"

define void @_start() #0 {
entry:
  ret void
}

attributes #0 = { "target-features"="+atomics,+bulk-memory,+mutable-globals,+sign-ext" }



