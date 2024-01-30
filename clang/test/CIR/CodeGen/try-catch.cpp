// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

double division(int a, int b);

// CHECK: cir.func @_Z2tcv()
unsigned long long tc() {
  int x = 50, y = 3;
  unsigned long long z;

  try {
    // CHECK: cir.scope {
    // CHECK: %[[msg:.*]] = cir.alloca !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>, ["msg"]
    // CHECK: %[[idx:.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["idx"]

    // CHECK: %[[try_eh:.*]] = cir.try {
    // CHECK: %[[eh_info:.*]] = cir.alloca !cir.ptr<!cir.eh.info>, cir.ptr <!cir.ptr<!cir.eh.info>>, ["__exception_ptr"]
    // CHECK: %[[local_a:.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["a", init]
    int a = 4;
    z = division(x, y);
    // CHECK: %[[div_res:.*]] = cir.try_call exception(%[[eh_info]]) @_Z8divisionii({{.*}}) : (!cir.ptr<!cir.ptr<!cir.eh.info>>, !s32i, !s32i) -> f64
    a++;

  // CHECK: cir.catch(%[[try_eh]] : !cir.ptr<!cir.eh.info>, [
  } catch (int idx) {
    // CHECK: type (#cir.global_view<@_ZTIi> : !cir.ptr<!u8i>)
    // CHECK: {
    // CHECK:   %[[catch_idx_addr:.*]] = cir.catch_param(%[[try_eh]]) -> !cir.ptr<!s32i>
    // CHECK:   %[[idx_load:.*]] = cir.load %[[catch_idx_addr]] : cir.ptr <!s32i>, !s32i
    // CHECK:   cir.store %[[idx_load]], %[[idx]] : !s32i, cir.ptr <!s32i> loc(#loc25)
    z = 98;
    idx++;
  } catch (const char* msg) {
    // CHECK: type (#cir.global_view<@_ZTIPKc> : !cir.ptr<!u8i>)
    // CHECK: {
    // CHECK:   %[[msg_addr:.*]] = cir.catch_param(%[[try_eh]]) -> !cir.ptr<!s8i> loc(#loc37)
    // CHECK:   cir.store %[[msg_addr]], %[[msg]] : !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>> loc(#loc37)
    z = 99;
    (void)msg[0];
  }

  return z;
}