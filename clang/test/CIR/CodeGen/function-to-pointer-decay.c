// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

void f(void);

void test_call_lvalue_cast() {
  (*(void (*)(int))f)(42);
}

// CHECK: cir.func {{.*}}@test_call_lvalue_cast()
// CHECK: [[F:%.+]] = cir.get_global @f
// CHECK: [[CASTED:%.+]] = cir.cast(bitcast, [[F]]
// CHECK: [[CONST:%.+]] = cir.const #cir.int<42>
// CHECK: cir.call [[CASTED]]([[CONST]])
