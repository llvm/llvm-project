// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Foo {
  void m1(int);
  virtual void m2(int);
  virtual void m3(int);
};

auto make_non_virtual() -> void (Foo::*)(int) {
  return &Foo::m1;
}

// CHECK-LABEL: cir.func @_Z16make_non_virtualv() -> !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK:   %{{.+}} = cir.const #cir.method<@_ZN3Foo2m1Ei> : !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK: }

auto make_virtual() -> void (Foo::*)(int) {
  return &Foo::m3;
}

// CHECK-LABEL: cir.func @_Z12make_virtualv() -> !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK:   %{{.+}} = cir.const #cir.method<vtable_offset = 8> : !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK: }

auto make_null() -> void (Foo::*)(int) {
  return nullptr;
}

// CHECK-LABEL: cir.func @_Z9make_nullv() -> !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK:   %{{.+}} = cir.const #cir.method<null> : !cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>
//       CHECK: }

void call(Foo *obj, void (Foo::*func)(int), int arg) {
  (obj->*func)(arg);
}

// CHECK-LABEL: cir.func @_Z4callP3FooMS_FviEi
//       CHECK:   %[[CALLEE:.+]], %[[THIS:.+]] = cir.get_method %{{.+}}, %{{.+}} : (!cir.method<!cir.func<!void (!s32i)> in !ty_22Foo22>, !cir.ptr<!ty_22Foo22>) -> (!cir.ptr<!cir.func<!void (!cir.ptr<!void>, !s32i)>>, !cir.ptr<!void>)
//  CHECK-NEXT:   %[[#ARG:]] = cir.load %{{.+}} : !cir.ptr<!s32i>, !s32i
//  CHECK-NEXT:   cir.call %[[CALLEE]](%[[THIS]], %[[#ARG]]) : (!cir.ptr<!cir.func<!void (!cir.ptr<!void>, !s32i)>>, !cir.ptr<!void>, !s32i) -> ()
//       CHECK: }
