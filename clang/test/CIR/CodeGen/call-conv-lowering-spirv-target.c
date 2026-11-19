// Checks which SPIR-V triples run CallConvLowering in the Clang pipeline. The
// SPIR-V ABI extends a char argument with signext, so it is there only when the
// pass runs.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple spirv32-unknown-unknown -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple spirv64-intel -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON

// ON: define {{.*}}void @take_char(i8 noundef signext %{{[^,)]+}})
// OFF: define {{.*}}void @take_char(i8 noundef %{{[^,)]+}})
void take_char(char c) {}
