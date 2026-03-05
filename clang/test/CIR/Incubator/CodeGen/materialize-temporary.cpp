// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test MaterializeTemporaryExpr when binding const reference to rvalue
int get_value() { return 42; }

void test_const_ref_binding() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_const_ref_bindingv
  const int &x = 5;
  // CHECK: %{{.*}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init]
  // CHECK: %{{.*}} = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init, const]
  // CHECK: cir.scope {
  // CHECK: %{{.*}} = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store {{.*}} %{{.*}}, %{{.*}} : !s32i, !cir.ptr<!s32i>
  // CHECK: }
}

void test_const_ref_expr() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_const_ref_exprv
  const int &y = get_value();
  // CHECK: %{{.*}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init]
  // CHECK: %{{.*}} = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["y", init, const]
  // CHECK: cir.scope {
  // CHECK: %{{.*}} = cir.call @{{.*}}get_valuev()
  // CHECK: }
}

void test_const_ref_arithmetic() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_const_ref_arithmeticv
  int a = 10;
  const int &z = a + 5;
  // CHECK: %{{.*}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0", init]
  // CHECK: %{{.*}} = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["z", init, const]
  // CHECK: cir.scope {
  // CHECK: %{{.*}} = cir.load {{.*}} %{{.*}}
  // CHECK: %{{.*}} = cir.const #cir.int<5> : !s32i
  // CHECK: %{{.*}} = cir.binop(add, %{{.*}}, %{{.*}})
  // CHECK: }
}

struct S {
  int val;
  S(int v) : val(v) {}
};

S make_s() { return S(100); }

void test_const_ref_struct() {
  // CHECK-LABEL: cir.func{{.*}} @{{.*}}test_const_ref_structv
  const S &s = make_s();
  // Temporary S object should be materialized
  // CHECK: %{{.*}} = cir.alloca {{.*}}, !cir.ptr<{{.*}}rec_S{{.*}}>, ["ref.tmp0"]
  // CHECK: %{{.*}} = cir.alloca !cir.ptr<{{.*}}>, !cir.ptr<!cir.ptr<{{.*}}>>, ["s", init, const]
}
