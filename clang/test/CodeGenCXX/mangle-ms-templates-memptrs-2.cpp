// RUN: %clang_cc1 -Wno-microsoft -fms-extensions -fno-rtti -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

template <typename T, int (T::*)() = nullptr>
struct J {};

template <typename T, int T::* = nullptr>
struct K {};

struct __single_inheritance M;
J<M> m;
// CHECK-DAG: @"?m@@3U?$J@UM@@$0A@@@A"

K<M> m2;
// CHECK-DAG: @"?m2@@3U?$K@UM@@$0?0@@A"

struct __multiple_inheritance N;
J<N> n;
// CHECK-DAG: @"?n@@3U?$J@UN@@$HA@@@A"

K<N> n2;
// CHECK-DAG: @"?n2@@3U?$K@UN@@$0?0@@A"

struct __virtual_inheritance O;
J<O> o;
// CHECK-DAG: @"?o@@3U?$J@UO@@$IA@A@@@A"

K<O> o2;
// CHECK-DAG: @"?o2@@3U?$K@UO@@$FA@?0@@A"

struct P;
J<P> p;
// CHECK-DAG: @"?p@@3U?$J@UP@@$JA@A@?0@@A"

K<P> p2;
// CHECK-DAG: @"?p2@@3U?$K@UP@@$GA@A@?0@@A"

#pragma pointers_to_members(full_generality, virtual_inheritance)

struct S {
  int a, b;
  void f();
  virtual void g();
};

struct GeneralBase {
  virtual void h();
};
struct MostGeneral : S, virtual GeneralBase {
  virtual void h();
};
template <void (MostGeneral::*MP)()>
struct ClassTemplate {
  ClassTemplate() {}
};

template struct ClassTemplate<&MostGeneral::h>;

// Test that we mangle in the vbptr offset, which is 12 here.
//
// CHECK: define weak_odr dso_local x86_thiscallcc noundef %struct.ClassTemplate* @"??0?$ClassTemplate@$J??_9MostGeneral@@$BA@AEA@M@3@@QAE@XZ"
