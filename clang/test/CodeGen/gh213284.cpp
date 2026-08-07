// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

struct Ref {
  unsigned long long bits;
};

template <typename>
struct Result {
  Result() : thing(0) {}
  Ref thing;
};

Result<void> construct() {
  return Result<void>();
}
// CHECK-LABEL: define {{.*}}construct
// CHECK: store i64 0
