; REQUIRES: x86
;
; RUN: cd %T
; RUN: opt -module-summary %s -o obj1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o obj2.o
; RUN: opt -module-summary %s -o %t_obj3.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t_obj4.o
;
; Objects with a relative path.
; RUN: rm -f *.lto.o *.s
; RUN: ld.lld --lto-output-module-name --save-temps=prelink --lto-obj-path=aaa --lto-emit-asm --thinlto-jobs=1 --entry=f obj1.o obj2.o -o bin1
; RUN: ls -1 *.lto.o *.s | FileCheck %s --check-prefixes=OBJPATHOUT1,PRELINKOUT1
; With thinlto-jobs=all.
; RUN: rm -f *.lto.o *.s
; RUN: ld.lld --lto-output-module-name --save-temps=prelink --lto-obj-path=aaa --lto-emit-asm --thinlto-jobs=all --entry=f obj1.o obj2.o -o bin1
; RUN: ls -1 *.lto.o *.s | FileCheck %s --check-prefixes=OBJPATHOUT1,PRELINKOUT1
; Objects with an absolute path.
; RUN: rm -f *.lto.o *.s
; RUN: ld.lld --lto-output-module-name --save-temps=prelink --lto-obj-path=aaa --lto-emit-asm --thinlto-jobs=1 --entry=f %t_obj3.o %t_obj4.o -o bin2
; RUN: ls -1 *.lto.o *.s | FileCheck %s --check-prefixes=OBJPATHOUT2,PRELINKOUT2
; Objects in an archive
; RUN: rm -f *.lto.o *.s
; RUN: llvm-ar rcS ar.a obj1.o obj2.o
; RUN: ld.lld --lto-output-module-name --save-temps=prelink --lto-obj-path=aaa --lto-emit-asm --thinlto-jobs=1 --entry=f ar.a -o bin1
; RUN: ls -1 *.lto.o *.s | FileCheck %s --check-prefixes=OBJPATHOUT3,PRELINKOUT3
; Use with thinlto-cahce
; RUN: rm -f *.lto.o *.s
; RUN: ld.lld --lto-output-module-name --save-temps=prelink --thinlto-cache-dir=thinlto-cache --lto-emit-asm --lto-obj-path=aaa --thinlto-jobs=1 --entry=f obj1.o obj2.o -o bin1
; RUN: ls -1 *.lto.o *.s | FileCheck %s --check-prefixes=OBJPATHOUT1,PRELINKOUT1
;
; OBJPATHOUT1-DAG: aaa_1_obj1.lto.o
; OBJPATHOUT1-DAG: aaa_2_obj2.lto.o
; PRELINKOUT1-DAG: bin1_1_obj1.lto.o
; PRELINKOUT1-DAG: bin1_2_obj2.lto.o
; PRELINKOUT1-DAG: bin1_1_obj1.lto.s
; PRELINKOUT1-DAG: bin1_2_obj2.lto.s
; OBJPATHOUT2-DAG: aaa_1_{{.*}}_obj3.lto.o
; OBJPATHOUT2-DAG: aaa_2_{{.*}}obj4.lto.o
; PRELINKOUT2-DAG: bin2_1_{{.*}}obj3.lto.o
; PRELINKOUT2-DAG: bin2_2_{{.*}}obj4.lto.o
; PRELINKOUT2-DAG: bin2_1_{{.*}}obj3.lto.s
; PRELINKOUT2-DAG: bin2_2_{{.*}}obj4.lto.s
; OBJPATHOUT3-DAG: aaa_1_ar.a_obj1.lto.o
; OBJPATHOUT3-DAG: aaa_2_ar.a_obj2.lto.o
; PRELINKOUT3-DAG: bin1_1_ar.a_obj1.lto.o
; PRELINKOUT3-DAG: bin1_2_ar.a_obj2.lto.o
; PRELINKOUT3-DAG: bin1_1_ar.a_obj1.lto.s
; PRELINKOUT3-DAG: bin1_2_ar.a_obj2.lto.s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
