; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: llvm-mc %p/Inputs/unnamed-addr-lib.s -o %t2.o -filetype=obj -triple=x86_64-pc-linux
; RUN: ld.lld %t2.o -shared -o %t2.so
; RUN: ld.lld %t.o %t2.so -o %t.so -save-temps -shared
; RUN: llvm-dis %t.so.0.2.internalize.bc -o - | FileCheck %s

; CHECK: @foo = weak_odr unnamed_addr constant i8 42
; CHECK: @bar = weak_odr unnamed_addr constant i8 42

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = linkonce_odr unnamed_addr constant i8 42
@bar = linkonce_odr unnamed_addr constant i8 42
