// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -clangir-disable-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -clangir-enable-call-conv-lowering -clangir-disable-call-conv-lowering \
// RUN:     -clangir-enable-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - \
// RUN:     -clangir-disable-call-conv-lowering -clangir-enable-call-conv-lowering \
// RUN:     -clangir-disable-call-conv-lowering \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=ON

typedef struct { long a, b, c, d; } Big;

// A 32-byte record returns through memory once the pass runs, and first-class
// otherwise, so the signature alone says whether the pass ran.  The ON
// signature is also what classic CodeGen emits, which the last RUN line pins.
Big ret_big(long a) {
  Big b = {a, a, a, a};
  return b;
}

// ON: define dso_local void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{[^,)]+}}, i64 noundef %{{[^,)]+}})
// OFF: define dso_local %struct.Big @ret_big(i64 noundef %{{[^,)]+}})
