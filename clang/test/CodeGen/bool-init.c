// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm < %s | FileCheck %s

// Check that the type of this global isn't i1
// CHECK: @test ={{.*}} global i8 1
_Bool test = &test;
