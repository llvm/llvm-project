;; Wrapped symbols are marked as weak for LTO to inhibit IPO. The ARM backend
;; was previously explicitly marking the personality function as global, so if
;; we attempted to wrap the personality function, its initial binding would be
;; weak and then the ARM backend would attempt to change it to global, causing
;; an error. Verify that the ARM backend no longer does this and we can
;; successfully wrap the personality symbol.

; REQUIRES: arm

; RUN: llvm-as -o %t.bc %s
; RUN: ld.lld -shared --wrap __gxx_personality_v0 %t.bc -o %t.so
; RUN: llvm-readelf --dyn-syms %t.so | FileCheck %s

; CHECK: GLOBAL {{.*}} __wrap___gxx_personality_v0
;; This should be GLOBAL when PR52004 is fixed.
; CHECK: WEAK {{.*}} __gxx_personality_v0

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-linux-android"

define i32 @__gxx_personality_v0(...) {
  ret i32 0
}

define void @dummy() optnone noinline personality ptr @__gxx_personality_v0 {
  invoke void @dummy() to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}
