// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -emit-llvm -fmodule-local-hints -o - %t/main.cpp | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++20 -emit-llvm -o - %t/main.cpp | FileCheck %s --check-prefix=NOHINTS

//--- hdr.h
struct HdrStruct {};
template <class T> int hdr_size(T) { return sizeof(T); }
template <class F> int hdr_call(F f) { return f(); }
template <int (*F)()> int hdr_callp() { return F(); }
template <template <class> class C> int hdr_tt() { return sizeof(C<int>); }
template <class... Ts> int hdr_pack(Ts...) { return sizeof...(Ts); }
template <class T> struct HdrBox {
  T t;
  int get() { return sizeof(t); }
};

//--- main.cpp
#include "hdr.h"

struct Local {};
int local_fn() { return 1; }
template <class T> struct LocalBox { T t; };
inline int main_inline() { return hdr_call([] { return 2; }); }

int use() {
  return hdr_size(Local()) + hdr_size((Local *)nullptr) +
         hdr_size(HdrBox<Local>()) + main_inline() + hdr_callp<local_fn>() +
         hdr_tt<LocalBox>() + hdr_pack(1, Local()) + HdrBox<Local>().get() +
         hdr_size(HdrStruct()) + HdrBox<int>().get();
}

// A plain function, never marked.
// CHECK: define {{.*}} @_Z8local_fnv() #[[OFF:[0-9]+]]

// The inline function defined in main.cpp is marked either way.
// CHECK-DAG: define linkonce_odr {{.*}} @_Z11main_inlinev() #[[ON:[0-9]+]]

// Templates defined in hdr.h, instantiated with an entity from main.cpp: a
// class type, a pointer to one, a header class template specialized with one,
// a lambda closure type, a function, a class template, and a pack element.
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_sizeI5LocalEiT_({{.*}}) #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_sizeIP5LocalEiT_({{.*}}) #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_sizeI6HdrBoxI5LocalEEiT_({{.*}}) #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_callIZ11main_inlinevEUlvE_EiT_({{.*}}) #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z9hdr_callpIXadL_Z8local_fnvEEEiv() #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z6hdr_ttI8LocalBoxEiv() #[[ON]]
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_packIJi5LocalEEiDpT_({{.*}}) #[[ON]]

// A member function of a header class template specialized with a main.cpp
// class.
// CHECK-DAG: define linkonce_odr {{.*}} @_ZN6HdrBoxI5LocalE3getEv({{.*}}) #[[ON]]

// Instantiations with only header and builtin types are never marked.
// CHECK-DAG: define linkonce_odr {{.*}} @_Z8hdr_sizeI9HdrStructEiT_({{.*}}) #[[OFF]]
// CHECK-DAG: define linkonce_odr {{.*}} @_ZN6HdrBoxIiE3getEv({{.*}}) #[[OFF]]

// CHECK: attributes #[[OFF]] = {
// CHECK-NOT: likely_module_local
// CHECK-SAME: }
// CHECK: attributes #[[ON]] = { {{.*}}likely_module_local{{.*}} }

// NOHINTS-NOT: likely_module_local
