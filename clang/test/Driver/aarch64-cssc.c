// Test that target feature cssc is implemented and available correctly
// FEAT_CSSC is a required part of v8.9a/v9.4a and optional from v8.7a/v9.3a onwards.
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf                         %s 2>&1 | FileCheck %s --check-prefix=ABSENT_CSSC
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.7-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.9-a        %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.9-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv8.9-a+nocssc %s 2>&1 | FileCheck %s --check-prefix=NO_CSSC
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.2-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.4-a        %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.4-a+cssc   %s 2>&1 | FileCheck %s
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -march=armv9.4-a+nocssc %s 2>&1 | FileCheck %s --check-prefix=NO_CSSC
// RUN: %clang -S -o - -emit-llvm --target=aarch64-none-elf -mcpu=ampere1b          %s 2>&1 | FileCheck %s

// CHECK: "target-features"="{{.*}},+cssc
// NO_CSSC: "target-features"="{{.*}},-cssc
// ABSENT_CSSC-NOT: "target-features"="{{.*}},+cssc
// ABSENT_CSSC-NOT: "target-features"="{{.*}},-cssc
void test() {}
