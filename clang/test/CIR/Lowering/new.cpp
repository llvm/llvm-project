// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void t_new_constant_size() {
  auto p = new double[16];
}

// LLVM: @_Z19t_new_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 128)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8

void t_new_multidim_constant_size() {
  auto p = new double[2][3][4];
}

// LLVM: @_Z28t_new_multidim_constant_sizev()
// LLVM:   %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ADDR:.*]] = call ptr @_Znam(i64 192)
// LLVM:   store ptr %[[ADDR]], ptr %[[ALLOCA]], align 8