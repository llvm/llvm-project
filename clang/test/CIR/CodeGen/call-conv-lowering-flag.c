// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -fno-clangir-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -fclangir-call-conv-lowering -fno-clangir-call-conv-lowering \
// RUN:     -fclangir-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -fno-clangir-call-conv-lowering -fclangir-call-conv-lowering \
// RUN:     -fno-clangir-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON

typedef struct { long a, b, c, d; } Big;

Big ret_big(long a) {
  Big b = {a, a, a, a};
  return b;
}

// ON: define dso_local void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}, i64 noundef %{{[^,)]+}})
// OFF: define dso_local %struct.Big @ret_big(i64 noundef %{{[^,)]+}})
