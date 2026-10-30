// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

consteval const int &get_ref(const int &x) {
  return x;
}

static constexpr int val = 10;

// CIR-LABEL: cir.func {{.*}} @_Z8test_refv
// OGCG-LABEL: define {{.*}} @_Z8test_refv
const int &test_ref() {
  // CIR: %[[R_ADDR:.*]] = cir.alloca "r" align(8) init const : !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[VAL_ADDR:.*]] = cir.const #cir.global_view<@_ZL3val> : !cir.ptr<!s32i>
  // CIR: cir.store align(8) %[[VAL_ADDR]], %[[R_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

  // OGCG: store ptr @_ZL3val, ptr %{{.*}}, align 8
  const int &r = get_ref(val);
  return r;
}
