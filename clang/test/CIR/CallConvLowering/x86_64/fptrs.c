// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -fclangir-call-conv-lowering %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-call-conv-lowering %s -o -| FileCheck %s -check-prefix=LLVM

typedef struct {
  int a;
} S;

typedef int (*myfptr)(S);

int foo(S s) { return 42 + s.a; }

// CHECK: cir.func {{.*@bar}}
// CHECK:   %[[#V0:]] = cir.alloca !cir.ptr<!cir.func<(!ty_S) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!ty_S) -> !s32i>>>, ["a", init]
// CHECK:   %[[#V1:]] = cir.get_global @foo : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// CHECK:   %[[#V2:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!cir.func<(!s32i) -> !s32i>>), !cir.ptr<!cir.func<(!ty_S) -> !s32i>>
// CHECK:   cir.store %[[#V2]], %[[#V0]] : !cir.ptr<!cir.func<(!ty_S) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!ty_S) -> !s32i>>>
void bar() {
  myfptr a = foo;
}

// CHECK: cir.func {{.*@baz}}(%arg0: !s32i
// CHECK:   %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, [""] {alignment = 4 : i64}
// CHECK:   %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!s32i>
// CHECK:   cir.store %arg0, %[[#V1]] : !s32i, !cir.ptr<!s32i>
// CHECK:   %[[#V2:]] = cir.alloca !cir.ptr<!cir.func<(!ty_S) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!ty_S) -> !s32i>>>, ["a", init]
// CHECK:   %[[#V3:]] = cir.get_global @foo : !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// CHECK:   %[[#V4:]] = cir.cast(bitcast, %[[#V3]] : !cir.ptr<!cir.func<(!s32i) -> !s32i>>), !cir.ptr<!cir.func<(!ty_S) -> !s32i>>
// CHECK:   cir.store %[[#V4]], %[[#V2]] : !cir.ptr<!cir.func<(!ty_S) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!ty_S) -> !s32i>>>
// CHECK:   %[[#V5:]] = cir.load %[[#V2]] : !cir.ptr<!cir.ptr<!cir.func<(!ty_S) -> !s32i>>>, !cir.ptr<!cir.func<(!ty_S) -> !s32i>>
// CHECK:   %[[#V6:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!s32i>
// CHECK:   %[[#V7:]] = cir.load %[[#V6]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[#V8:]] = cir.cast(bitcast, %[[#V5]] : !cir.ptr<!cir.func<(!ty_S) -> !s32i>>), !cir.ptr<!cir.func<(!s32i) -> !s32i>>
// CHECK:   %[[#V9:]] = cir.call %[[#V8]](%[[#V7]]) : (!cir.ptr<!cir.func<(!s32i) -> !s32i>>, !s32i) -> !s32i

// LLVM: define dso_local void @baz(i32 %0)
// LLVM:   %[[#V1:]] = alloca %struct.S, i64 1
// LLVM:   store i32 %0, ptr %[[#V1]]
// LLVM:   %[[#V2:]] = alloca ptr, i64 1
// LLVM:   store ptr @foo, ptr %[[#V2]]
// LLVM:   %[[#V3:]] = load ptr, ptr %[[#V2]]
// LLVM:   %[[#V4:]] = load i32, ptr %[[#V1]]
// LLVM:   %[[#V5:]] = call i32 %[[#V3]](i32 %[[#V4]])

void baz(S s) {
  myfptr a = foo;
  a(s);
}
