// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir

int printf(const char *str, ...);

// CHECK: cir.func {{.*@bar}}
// CHECK:   %[[#V1:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CHECK:   %[[#V2:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CHECK:   cir.store %arg0, %[[#V0]] : !s32i, !cir.ptr<!s32i>
// CHECK:   cir.store %arg1, %[[#V1]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[#V2:]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 7>>
// CHECK:   %[[#V3:]] = cir.cast array_to_ptrdecay %[[#V2]] : !cir.ptr<!cir.array<!s8i x 7>> -> !cir.ptr<!s8i>
// CHECK:   %[[#V4:]] = cir.load %[[#V1]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[#V5:]] = cir.load %[[#V2]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[#V6:]] = cir.call @printf(%[[#V3]], %[[#V4]], %[[#V5]]) : (!cir.ptr<!s8i>, !s32i, !s32i) -> !s32i
void bar(int a, int b) {
  printf("%d %d\n", a, b);
}
