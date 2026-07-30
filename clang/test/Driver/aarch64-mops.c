// Test that target feature mops is implemented and available correctly
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.7-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.8-a        %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.8-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.8-a+nomops %s 2>&1 | FileCheck %s --check-prefix=NO_MOPS
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.2-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.3-a        %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.3-a+mops   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.3-a+nomops %s 2>&1 | FileCheck %s --check-prefix=NO_MOPS

// CHECK: "target-features"="{{.*}},+mops
// NO_MOPS: "target-features"="{{.*}},-mops

void test() {}
