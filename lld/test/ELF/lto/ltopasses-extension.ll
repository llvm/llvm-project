; REQUIRES: x86, plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; RUN: opt %s -o %t.o
; RUN: opt -module-summary %s -o %t_thin.o

; RUN: ld.lld -%loadnewpmbye --lto-newpm-passes="goodbye" -mllvm=%loadbye -mllvm=-wave-goodbye %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: Bye

; Entry-points in pipeline for regular/monolithic LTO
;
; RUN: ld.lld -%loadnewpmbye -mllvm=%loadbye -mllvm=-print-ep-callbacks %t.o \
; RUN:         -shared -o /dev/null | FileCheck --check-prefix=REGULAR %s
;
; REGULAR-NOT: PipelineStart
; REGULAR-NOT: PipelineEarlySimplification
; REGULAR-NOT: Peephole
; REGULAR-NOT: ScalarOptimizerLate
; REGULAR-NOT: Vectorizer
; REGULAR-NOT: Optimizer
;
; REGULAR: FullLinkTimeOptimizationEarly
; REGULAR: FullLinkTimeOptimizationLast

; Entry-points in Thin-LTO pipeline
;
; RUN: ld.lld -%loadnewpmbye -mllvm=%loadbye -mllvm=-print-ep-callbacks %t_thin.o \
; RUN:         -shared -o /dev/null | FileCheck --check-prefix=THIN %s
;
; THIN-NOT: FullLinkTimeOptimizationEarly
; THIN-NOT: FullLinkTimeOptimizationLast
; THIN-NOT: PipelineStart
;
; THIN: PipelineEarlySimplification
; THIN: Peephole
; THIN: ScalarOptimizerLate
; THIN: Peephole
; THIN: OptimizerEarly
; THIN: VectorizerStart
; THIN: VectorizerEnd
; THIN: OptimizerLast

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}
