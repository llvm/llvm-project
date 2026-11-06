// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -emit-module-interface %t/m.cppm -o %t/m.pcm
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -fmodule-file=m=%t/m.pcm -emit-cir %t/use.cpp -o %t/use-O1.cir
// RUN: FileCheck --check-prefix=CIR --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O1.cir %t/use.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -disable-llvm-passes -fclangir -fmodule-file=m=%t/m.pcm -emit-llvm %t/use.cpp -o %t/use-O1-cir.ll
// RUN: FileCheck --check-prefix=LLVM --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O1-cir.ll %t/use.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -disable-llvm-passes -fmodule-file=m=%t/m.pcm -emit-llvm %t/use.cpp -o %t/use-O1.ll
// RUN: FileCheck --check-prefix=LLVM --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O1.ll %t/use.cpp
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O0 -fclangir -fmodule-file=m=%t/m.pcm -emit-cir %t/use.cpp -o %t/use-O0.cir
// RUN: FileCheck --check-prefix=CIR --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O0.cir %t/use.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O0 -disable-llvm-passes -fclangir -fmodule-file=m=%t/m.pcm -emit-llvm %t/use.cpp -o %t/use-O0-cir.ll
// RUN: FileCheck --check-prefix=LLVM --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O0-cir.ll %t/use.cpp
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O0 -disable-llvm-passes -fmodule-file=m=%t/m.pcm -emit-llvm %t/use.cpp -o %t/use-O0.ll
// RUN: FileCheck --check-prefix=LLVM --implicit-check-not=_ZN1GIiED2Ev --input-file=%t/use-O0.ll %t/use.cpp
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -fclangir -emit-cir %t/m.cppm -o %t/m.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t/m.cir %t/m.cppm
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -disable-llvm-passes -fclangir -emit-llvm %t/m.cppm -o %t/m-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t/m-cir.ll %t/m.cppm
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O1 -disable-llvm-passes -emit-llvm %t/m.cppm -o %t/m.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t/m.ll %t/m.cppm

//--- g.h
template <class T> struct G {
  int get() { return v; }
  __attribute__((always_inline)) int getAlwaysInline() { return v; }
  ~G() { v = 0; }
  int v;
};
extern template struct G<int>;

//--- m.cppm
module;
#include "g.h"
export module m;
export using ::G;

export int plain() { return 1; }
export __attribute__((noinline)) int plainNoinline() { return 2; }
export __attribute__((always_inline)) int plainAlwaysInline() { return 3; }

export template <class T> struct S {
  int get();
  __attribute__((always_inline)) int getAlwaysInline();
};
template <class T> inline int S<T>::get() { return 4; }
template <class T> inline int S<T>::getAlwaysInline() { return 5; }
extern template struct S<int>;

export template <class T> struct D {
  __attribute__((always_inline)) D();
  __attribute__((always_inline)) ~D();
  int v;
};
template <class T> inline D<T>::D() : v(1) {}
template <class T> inline D<T>::~D() { v = 0; }
extern template struct D<int>;

// The owning module keeps the body of an extern template member.
// CIR: cir.func inline_hint alignment(2) available_externally @_ZNW1m1SIiE3getEv(%{{.+}}: !cir.ptr<!rec_S3Cint3E> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) -> (!s32i {llvm.noundef}) attributes {{{.*}}} {
// LLVM: define available_externally noundef i32 @_ZNW1m1SIiE3getEv(ptr noundef nonnull align 1 dereferenceable(1) %{{.+}})
export int useInModule() {
  S<int> s;
  return s.get();
}

//--- use.cpp
import m;

// Global module fragment extern template: only always_inline keeps its body.
// CIR-DAG: cir.func private @_ZN1GIiE3getEv(!cir.ptr<!rec_G3Cint3E> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZN1GIiE15getAlwaysInlineEv(%{{.+}}: !cir.ptr<!rec_G3Cint3E> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) -> (!s32i {llvm.noundef}) attributes {{{.*}}} {
// CIR-DAG: cir.func private @_ZN1GIiED1Ev(!cir.ptr<!rec_G3Cint3E> {llvm.align = 4 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef}) func_info<#cir.cxx_dtor<!rec_G3Cint3E>> attributes {{{.*}}}{{[^{]*}}{{$}}
// LLVM-DAG: declare noundef i32 @_ZN1GIiE3getEv(ptr noundef nonnull align 4 dereferenceable(4))
// LLVM-DAG: define available_externally noundef i32 @_ZN1GIiE15getAlwaysInlineEv(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}})
// LLVM-DAG: declare void @_ZN1GIiED1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4))

// Exported non-template functions are declared, even the always_inline one.
// CIR-DAG: cir.func private @_ZW1m5plainv() -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// CIR-DAG: cir.func private @_ZW1m13plainNoinlinev() -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// CIR-DAG: cir.func private @_ZW1m17plainAlwaysInlinev() -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// CIR-DAG: cir.func private @_ZW1m11useInModulev() -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// LLVM-DAG: declare noundef i32 @_ZW1m5plainv()
// LLVM-DAG: declare noundef i32 @_ZW1m13plainNoinlinev()
// LLVM-DAG: declare noundef i32 @_ZW1m17plainAlwaysInlinev()
// LLVM-DAG: declare noundef i32 @_ZW1m11useInModulev()

// Extern template members in the purview: only always_inline keeps its body.
// CIR-DAG: cir.func private @_ZNW1m1SIiE3getEv(!cir.ptr<!rec_S3Cint3E> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) -> (!s32i {llvm.noundef}) attributes {{{.*}}}{{[^{]*}}{{$}}
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZNW1m1SIiE15getAlwaysInlineEv(%{{.+}}: !cir.ptr<!rec_S3Cint3E> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) -> (!s32i {llvm.noundef}) attributes {{{.*}}} {
// LLVM-DAG: declare noundef i32 @_ZNW1m1SIiE3getEv(ptr noundef nonnull align 1 dereferenceable(1))
// LLVM-DAG: define available_externally noundef i32 @_ZNW1m1SIiE15getAlwaysInlineEv(ptr noundef nonnull align 1 dereferenceable(1) %{{.+}})

// Always_inline constructor and destructor variants keep their bodies.
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZNW1m1DIiEC1Ev(%{{.+}}: !cir.ptr<!rec_D3Cint3E> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_D3Cint3E, default>> attributes {{{.*}}} {
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZNW1m1DIiEC2Ev(%{{.+}}: !cir.ptr<!rec_D3Cint3E> {llvm.align = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_ctor<!rec_D3Cint3E, default>> attributes {{{.*}}} {
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZNW1m1DIiED1Ev(%{{.+}}: !cir.ptr<!rec_D3Cint3E> {llvm.align = 4 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_dtor<!rec_D3Cint3E>> attributes {{{.*}}} {
// CIR-DAG: cir.func always_inline alignment(2) available_externally @_ZNW1m1DIiED2Ev(%{{.+}}: !cir.ptr<!rec_D3Cint3E> {llvm.align = 4 : i64, llvm.dead_on_return = 4 : i64, llvm.dereferenceable = 4 : i64, llvm.nonnull, llvm.noundef} loc({{[^)]+}})) func_info<#cir.cxx_dtor<!rec_D3Cint3E>> attributes {{{.*}}} {
// LLVM-DAG: define available_externally void @_ZNW1m1DIiEC1Ev(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}})
// LLVM-DAG: define available_externally void @_ZNW1m1DIiEC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}})
// LLVM-DAG: define available_externally void @_ZNW1m1DIiED1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %{{.+}})
// LLVM-DAG: define available_externally void @_ZNW1m1DIiED2Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %{{.+}})
int use() {
  G<int> g;
  S<int> s;
  D<int> d;
  return g.get() + g.getAlwaysInline() + plain() + plainNoinline() +
         plainAlwaysInline() + s.get() + s.getAlwaysInline() + useInModule();
}
