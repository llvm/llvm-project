// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

consteval const int &get_ref(const int &x) {
  return x;
}

static constexpr int val = 10;

// CIR-LABEL: cir.func {{.*}} @_Z11test_returnv
// OGCG-LABEL: define {{.*}} @_Z11test_returnv
const int &test_return() {
  // CIR: %[[VAL_ADDR:.*]] = cir.const #cir.global_view<@_ZL3val> : !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[VAL_ADDR]], %[[RET_ADDR:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

  // OGCG: ret ptr @_ZL3val
  return get_ref(val);
}
