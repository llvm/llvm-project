// RUN: split-file %s %t

// Build the module interface to a PCM
// RUN: %clang_cc1 -std=c++20 -triple powerpc-ibm-aix \
// RUN:   -emit-module-interface %t/copymod.cppm -o %t/copymod.pcm

// Verify that module interface emits copyright global when compiled to IR

// RUN: %clang_cc1 -std=c++20 -triple powerpc-ibm-aix -emit-llvm %t/copymod.cppm -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-MOD
// CHECK-MOD: @__loadtime_comment_str = internal unnamed_addr constant [10 x i8] c"module me\00", section "__loadtime_comment"
// CHECK-MOD: @llvm.used = appending global {{.*}} @__loadtime_comment_str

// Compile an importing TU that uses the prebuilt module and verify that it
// does NOT re-emit the module's copyright global.

// RUN: %clang_cc1 -std=c++20 -triple powerpc-ibm-aix \
// RUN:   -fprebuilt-module-path=%t -emit-llvm %t/importmod.cc -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK-IMPORT
// CHECK-IMPORT-NOT: @__loadtime_comment_str
// CHECK-IMPORT-NOT: c"module me\00"

//--- copymod.cppm
export module copymod;
#pragma comment(copyright, "module me")
export inline void f() {}

//--- importmod.cc
import copymod;
void g() { f(); }
