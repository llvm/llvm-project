// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -emit-llvm -std=c++17 -o - %s | FileCheck %s

// Tests that exception specifiers are stripped when forming the
// mangled CFI type name.

void f() noexcept {}

// CHECK: define{{.*}} void @_Z1fv({{.*}} !type [[TS1:![0-9]+]] !type [[TS2:![0-9]+]]

// CHECK: [[TS1]] = !{i64 0, !"_ZTSFvvE"}
// CHECK: [[TS2]] = !{i64 0, !"_ZTSFvvE.generalized"}
