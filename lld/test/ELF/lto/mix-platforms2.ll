; REQUIRES: x86
; RUN: llvm-as %s -o %tx64.o
; RUN: llvm-as %S/Inputs/i386-empty.ll -o %ti386.o
; RUN: not ld.lld %ti386.o %tx64.o -o /dev/null 2>&1 | FileCheck %s

; CHECK: {{.*}}x64.o is incompatible with {{.*}}i386.o

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
