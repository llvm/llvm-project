// Test that target feature hbc is implemented and available correctly
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv8.7-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv8.8-a       %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv8.8-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv8.8-a+nohbc %s 2>&1 | FileCheck %s --check-prefix=NO_HBC
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv9.2-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv9.3-a       %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv9.3-a+hbc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm -target aarch64-none-none-eabi -march=armv9.3-a+nohbc %s 2>&1 | FileCheck %s --check-prefix=NO_HBC

// CHECK: "target-features"="{{.*}},+hbc
// NO_HBC: "target-features"="{{.*}},-hbc

void test() {}