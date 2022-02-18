; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
;; This test covers both FullLTO and ThinLTO code paths because we have observed
;; (unexpected) differences between the two.
; RUN: llvm-as %t/test.ll -o %t/test.o
; RUN: llvm-as %t/test2.ll -o %t/test2.o
; RUN: opt -module-summary %t/test.ll -o %t/test.thinlto.o
; RUN: opt -module-summary %t/test2.ll -o %t/test2.thinlto.o

; RUN: %lld -lSystem %t/test.o %t/test2.o -o %t/test
; RUN: llvm-nm -m %t/test | FileCheck %s --check-prefix=LTO

; RUN: %lld -lSystem -dylib %t/test.o %t/test2.o -o %t/test.dylib
; RUN: llvm-nm -m %t/test.dylib | FileCheck %s --check-prefix=LTO-DYLIB

; RUN: %lld -lSystem %t/test.thinlto.o %t/test2.o -o %t/test.thinlto
; RUN: llvm-nm -m %t/test.thinlto | FileCheck %s --check-prefix=THINLTO

; RUN: %lld -lSystem -dylib %t/test.thinlto.o %t/test2.o -o %t/test.thinlto.dylib
; RUN: llvm-nm -m %t/test.thinlto.dylib | FileCheck %s --check-prefix=THINLTO

; LTO-DAG: (__DATA,__data) non-external _global_unnamed
; LTO-DAG: (__DATA,__data) non-external _local_unnamed
;; LD64 marks this with (was a private external). IMO both LD64 and LLD should
;; mark all the other internalized symbols with (was a private external).
; LTO-DAG: (__TEXT,__const) non-external _local_unnamed_always_const
; LTO-DAG: (__TEXT,__const) non-external _local_unnamed_const
;; LD64 doesn't internalize this -- it emits it as a weak external -- which I
;; think is a missed optimization on its end.
; LTO-DAG: (__TEXT,__const) non-external _local_unnamed_sometimes_const

;; The output here is largely identical to LD64's, except that the non-external
;; symbols here are all marked as (was a private external) by LD64. LLD should
;; follow suit.
; LTO-DYLIB-DAG: (__DATA,__data) non-external _global_unnamed
; LTO-DYLIB-DAG: (__DATA,__data) weak external _local_unnamed
; LTO-DYLIB-DAG: (__TEXT,__const) non-external _local_unnamed_always_const
; LTO-DYLIB-DAG: (__TEXT,__const) non-external _local_unnamed_const
; LTO-DYLIB-DAG: (__TEXT,__const) weak external _local_unnamed_sometimes_const

; THINLTO-DAG: (__DATA,__data) non-external (was a private external) _global_unnamed
; THINLTO-DAG: (__DATA,__data) weak external _local_unnamed
;; The next two symbols are rendered as non-external (was a private external)
;; by LD64. This is a missed optimization on LLD's end.
; THINLTO-DAG: (__TEXT,__const) weak external _local_unnamed_always_const
; THINLTO-DAG: (__TEXT,__const) weak external _local_unnamed_const
;; LD64 actually fails to link when the following symbol is included in the test
;; input, instead producing this error:
;; reference to bitcode symbol '_local_unnamed_sometimes_const' which LTO has not compiled in '_used' from /tmp/lto.o for architecture x86_64
; THINLTO-DAG: (__TEXT,__const) weak external _local_unnamed_sometimes_const

;--- test.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@global_unnamed = linkonce_odr unnamed_addr global i8 42
@local_unnamed_const = linkonce_odr local_unnamed_addr constant i8 42
@local_unnamed_always_const = linkonce_odr local_unnamed_addr constant i8 42
@local_unnamed_sometimes_const = linkonce_odr local_unnamed_addr constant i8 42
@local_unnamed = linkonce_odr local_unnamed_addr global i8 42
@used = hidden constant [5 x i8*] [i8* @global_unnamed, i8* @local_unnamed,
  i8* @local_unnamed_const, i8* @local_unnamed_always_const,
  i8* @local_unnamed_sometimes_const]
@llvm.used = appending global [1 x [5 x i8*]*] [[5 x i8*]* @used]

define void @main() {
  ret void
}

;--- test2.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@local_unnamed_always_const = linkonce_odr local_unnamed_addr constant i8 42
@local_unnamed_sometimes_const = linkonce_odr local_unnamed_addr global i8 42
