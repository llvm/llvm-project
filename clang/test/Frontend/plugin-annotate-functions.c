// RUN: %clang -fplugin=%llvmshlibdir/AnnotateFunctions%pluginext -emit-llvm -S %s -o - | FileCheck %s
// REQUIRES: plugins, examples

// CHECK: [[STR_VAR:@.+]] = private unnamed_addr constant [19 x i8] c"example_annotation\00"
// CHECK: @llvm.global.annotations = {{.*}}@fn1{{.*}}[[STR_VAR]]{{.*}}@fn2{{.*}}[[STR_VAR]]
void fn1() { }
void fn2() { }
