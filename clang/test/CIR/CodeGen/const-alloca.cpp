// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int produce_int();
void blackbox(const int &);

void local_const_int() {
  const int x = produce_int();
}

// CHECK-LABEL: @_Z15local_const_intv
// CHECK:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CHECK: }

void param_const_int(const int x) {}

// CHECK-LABEL: @_Z15param_const_inti
// CHECK:  %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CHECK: }

void local_constexpr_int() {
  constexpr int x = 42;
  blackbox(x);
}

// CHECK-LABEL: @_Z19local_constexpr_intv
// CHECK:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CHECK: }

void local_reference() {
  int x = 0;
  int &r = x;
}

// CHECK-LABEL: @_Z15local_referencev
// CHECK:   %{{.+}} = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CHECK: }

struct Foo {
  int a;
  int b;
};

Foo produce_foo();

void local_const_struct() {
  const Foo x = produce_foo();
}

// CHECK-LABEL: @_Z18local_const_structv
// CHECK:   %{{.+}} = cir.alloca !ty_Foo, !cir.ptr<!ty_Foo>, ["x", init, const]
// CHECK: }
