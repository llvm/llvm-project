// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

typedef float V4F __attribute__((vector_size(16)));

struct Empty {};

extern "C" {

union VecAndEmpty {
  V4F v;
  Empty e;
};

void take_vec_and_empty(union VecAndEmpty u);
void call_vec_and_empty(union VecAndEmpty u) { take_vec_and_empty(u); }
// CHECK-DAG: declare void @take_vec_and_empty(<2 x double>)

union VecAndEmpty ret_vec_and_empty(void);
void call_ret_vec_and_empty(void) { ret_vec_and_empty(); }
// CHECK-DAG: declare <2 x double> @ret_vec_and_empty()

struct VecNoUniqueAddress {
  V4F v;
  [[no_unique_address]] Empty e;
};

void take_vec_nua(VecNoUniqueAddress s);
void call_vec_nua(VecNoUniqueAddress s) { take_vec_nua(s); }
// CHECK-DAG: declare void @take_vec_nua(<4 x float>)

VecNoUniqueAddress ret_vec_nua(void);
void call_ret_vec_nua(void) { ret_vec_nua(); }
// CHECK-DAG: declare <4 x float> @ret_vec_nua()
}
