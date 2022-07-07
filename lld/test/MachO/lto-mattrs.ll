; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

;; Verify that LTO behavior can be tweaked using -mattr.

; RUN: %lld -mcpu haswell -mllvm -mattr=+fma %t.o -o %t.dylib -dylib
; RUN: llvm-objdump -d --section="__text" --no-leading-addr --no-show-raw-insn %t.dylib | FileCheck %s --check-prefix=FMA

; RUN: %lld -mcpu haswell -mllvm -mattr=-fma %t.o -o %t.dylib -dylib
; RUN: llvm-objdump -d --section="__text" --no-leading-addr --no-show-raw-insn %t.dylib | FileCheck %s --check-prefix=NO-FMA

; FMA:      <_foo>:
; FMA-NEXT: vrcpss       %xmm0, %xmm0, %xmm1
; FMA-NEXT: vfmsub213ss  [[#]](%rip), %xmm1, %xmm0 ## xmm0 = (xmm1 * xmm0) - mem
; FMA-NEXT:                                        ## 0x2f4
; FMA-NEXT: vfnmadd132ss %xmm1, %xmm1, %xmm0
; FMA-NEXT: retq

; NO-FMA:      <_foo>:
; NO-FMA-NEXT: vrcpss %xmm0, %xmm0, %xmm1
; NO-FMA-NEXT: vmulss %xmm1, %xmm0, %xmm0
; NO-FMA-NEXT: vmovss [[#]](%rip), %xmm2  ## xmm2 =
; NO-FMA-NEXT:                            ## 0x
; NO-FMA-NEXT: vsubss %xmm0, %xmm2, %xmm0
; NO-FMA-NEXT: vmulss %xmm0, %xmm1, %xmm0
; NO-FMA-NEXT: vaddss %xmm0, %xmm1, %xmm0
; NO-FMA-NEXT: retq

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float %x) #0 {
  %div = fdiv fast float 1.0, %x
  ret float %div
}

attributes #0 = { "unsafe-fp-math"="true" "reciprocal-estimates"="divf,vec-divf" }
