// The ABI-free boundary round-trips through an external MLIR tool: emit
// pre-lowering CIR, transform it with cir-opt, then resume in cc1, which
// runs target/ABI lowering and produces working code.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -clangir-stop-before-lowering -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-canonicalize -o %t.opt.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.opt.cir \
// RUN:   -emit-llvm -o - | FileCheck %s

struct S {
  int a, b;
};

struct S make(int x) {
  struct S s = {x, x};
  return s;
}

// The resumed pipeline applies the x86_64 System V ABI: the 8-byte struct
// return is coerced to i64.
// CHECK: define {{.*}} i64 @make(i32 {{.*}}%{{.+}})
