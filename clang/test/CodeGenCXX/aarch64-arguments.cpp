// RUN: %clang_cc1 -triple arm64-none-linux -emit-llvm -w -o - %s | FileCheck -check-prefix=PCS %s

// PCS: define{{.*}} void @{{.*}}(i64 %a.coerce)
struct s0 {};
void f0(s0 a) {}
