// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-translate --cir-to-llvmir --disable-cc-lowering %t.cir -o %t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: grep '^target datalayout' %t.ll > %t.cir.dl
// RUN: grep '^target datalayout' %t.ogcg.ll > %t.ogcg.dl
// RUN: diff %t.cir.dl %t.ogcg.dl
// RUN: FileCheck %s -input-file %t.cir.dl
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -x ir %t.ll -o /dev/null

// The data layout CIRGen records in the module is the target's, so it comes
// back out of cir-translate spelled exactly as classic codegen spells it.

int foo(_Bool b) { return b; }

// CHECK: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
