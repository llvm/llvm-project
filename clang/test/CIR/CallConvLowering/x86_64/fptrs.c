// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -fclangir-call-conv-lowering %s -o - | FileCheck %s

typedef struct {
  int a;
} S;

typedef int (*myfptr)(S);

int foo(S s) { return 42 + s.a; }

// CHECK: cir.func {{.*@bar}}
// CHECK:   %[[#V0:]] = cir.alloca !cir.ptr<!cir.func<!s32i (!ty_S)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!ty_S)>>>, ["a", init]
// CHECK:   %[[#V1:]] = cir.get_global @foo : !cir.ptr<!cir.func<!s32i (!s32i)>> 
// CHECK:   %[[#V2:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!cir.func<!s32i (!s32i)>>), !cir.ptr<!cir.func<!s32i (!ty_S)>>
// CHECK:   cir.store %[[#V2]], %[[#V0]] : !cir.ptr<!cir.func<!s32i (!ty_S)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!ty_S)>>>
void bar() {
  myfptr a = foo;
}
