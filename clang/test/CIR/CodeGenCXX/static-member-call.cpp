// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
  static int static_func(int a);
};

// CIR-LABEL: cir.func {{.*}} @_Z18test_instance_call1S
// OGCG-LABEL: define {{.*}} @_Z18test_instance_call1S
int test_instance_call(S s) {
  // CIR: %[[ARG:.*]] = cir.const #cir.int<42> : !s32i
  // CIR: %[[CALL:.*]] = cir.call @_ZN1S11static_funcEi(%[[ARG]]) : (!s32i {{.*}}) -> (!s32i {{.*}})
  // CIR: cir.store %[[CALL]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[RES:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
  // CIR: cir.return %[[RES]] : !s32i

  // OGCG: %[[CALL_OGCG:.*]] = call noundef i32 @_ZN1S11static_funcEi(i32 noundef 42)
  // OGCG: ret i32 %[[CALL_OGCG]]
  return s.static_func(42);
}

// CIR-LABEL: cir.func {{.*}} @_Z15test_arrow_callP1S
// OGCG-LABEL: define {{.*}} @_Z15test_arrow_callP1S
int test_arrow_call(S *p) {
  // CIR: %[[ARG:.*]] = cir.const #cir.int<100> : !s32i
  // CIR: %[[CALL:.*]] = cir.call @_ZN1S11static_funcEi(%[[ARG]]) : (!s32i {{.*}}) -> (!s32i {{.*}})
  // CIR: cir.store %[[CALL]], %[[RET:.*]] : !s32i, !cir.ptr<!s32i>
  // CIR: %[[RES:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
  // CIR: cir.return %[[RES]] : !s32i

  // OGCG: %[[CALL_OGCG:.*]] = call noundef i32 @_ZN1S11static_funcEi(i32 noundef 100)
  // OGCG: ret i32 %[[CALL_OGCG]]
  return p->static_func(100);
}
