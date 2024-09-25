// RUN: %clang_cc1 -emit-llvm -o - -std=c2x %s | FileCheck %s

// C2x 6.7.2.5 EXAMPLE 5
unsigned long long vla_size(int n) {
// CHECK: vla_size

  return sizeof(
    typeof_unqual(char[n + 3])
  ); // execution-time sizeof, translation-time typeof operation
// CHECK: [[N_ADDR:%.*]] = alloca i32
// CHECK: store i32 {{%.*}} ptr [[N_ADDR]]
// CHECK: [[N:%.*]] = load i32, ptr [[N_ADDR]]
// CHECK: [[TEMP:%.*]] = add nsw i32 [[N]], 3
// CHECK: [[RET:%.*]] = zext i32 [[TEMP]] to i64
// CHECK: ret i64 [[RET]]
}

int main() {
  return (int)vla_size(10); // vla_size returns 13
}

