// RUN: %clang_cc1 -triple aarch64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -emit-llvm -o - %s | FileCheck %s

// CHECK: define dso_local void @f({{.*}} !type [[TYPE:![0-9]+]] !type [[TYPE_GENERALIZED:![0-9]+]]
void f(__builtin_va_list l) {}

// CHECK-DAG: [[TYPE]] = !{i64 0, !"_ZTSFvSt9__va_listE"}
// CHECK-DAG: [[TYPE_GENERALIZED]] = !{i64 0, !"_ZTSFvSt9__va_listE.generalized"}
