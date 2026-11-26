// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// CHECK-DAG: @_ZN1QI19__type_pack_elementE1vE = linkonce_odr global
// CHECK-DAG: @_ZTV1RI18__make_integer_seqE = linkonce_odr constant
// CHECK-DAG: @r = global %struct.R
// CHECK: define linkonce_odr void @_ZN1RI18__make_integer_seqE1fEv

template <template <unsigned long, class...> class> struct Q {
  static int v;
};
template <template <unsigned long, class...> class M> int Q<M>::v;
int *p = &Q<__type_pack_element>::v;

template <template <template <class T, T...> class, class T, T> class>
struct R {
  virtual void f() {}
};

R<__make_integer_seq> r;
