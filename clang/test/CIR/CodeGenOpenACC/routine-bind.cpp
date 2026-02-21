// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s
// FIXME: We should run this against Windows mangling as well at one point.

#pragma acc routine seq bind("BIND1")
void Func1(){}

void Func2(){}
#pragma acc routine(Func2) seq bind("BIND2")

#pragma acc routine seq device_type(nvidia) bind("BIND3")
void Func3(){}

void Func4(){}
#pragma acc routine(Func4) seq device_type(radeon) bind("BIND4")

#pragma acc routine seq device_type(nvidia, host) bind("BIND5_N") device_type(multicore) bind("BIND5_M")
void Func5(){}

void Func6(){}
#pragma acc routine(Func6) seq device_type(radeon) bind("BIND6_R") device_type(multicore, host) bind("BIND6_M")

#pragma acc routine seq bind(BIND7)
void Func7(int){}

void Func8(float){}
#pragma acc routine(Func8) seq bind(BIND8)

#pragma acc routine seq device_type(nvidia) bind(BIND9)
void Func9(int, float, short){}

struct S{};
struct U{};
struct V{};

void Func10(S){}
#pragma acc routine(Func10) seq device_type(radeon) bind(BIND10)

#pragma acc routine seq device_type(nvidia, host) bind(BIND11_NVH) device_type(multicore) bind(BIND11_MC)
void Func11(U*, V&, int){}

int Func12(U, V, int){ return 0; }
#pragma acc routine(Func12) seq device_type(radeon) bind(BIND12_R) device_type(multicore, host) bind(BIND12_MCH)

struct HasFuncs {
#pragma acc routine seq bind(MEM)
  int MemFunc(int, double, HasFuncs&, S){ return 0; }
#pragma acc routine seq bind(MEM)
  int ConstMemFunc(int, double, HasFuncs&, S) const { return 0; }
#pragma acc routine seq bind(MEM)
  int VolatileMemFunc(int, double, HasFuncs&, S) const volatile { return 0; }
#pragma acc routine seq bind(MEM)
  int RefMemFunc(int, double, HasFuncs&, S) const && { return 0; }
#pragma acc routine seq bind(STATICMEM)
  int StaticMemFunc(int, double, HasFuncs&, U*){ return 0; }
};

void hasLambdas() {
  HasFuncs hf;
  hf.MemFunc(1, 1.0, hf, S{});
  hf.ConstMemFunc(1, 1.0, hf, S{});
  static_cast<const volatile HasFuncs>(hf).VolatileMemFunc(1, 1.0, hf, S{});
  HasFuncs{}.RefMemFunc(1, 1.0, hf, S{});
  U u;
  hf.StaticMemFunc(1, 1.0, hf, &u);
  int i, j, k, l;
#pragma acc routine seq bind(LAMBDA1)
  auto Lambda = [](int, float, double){};
#pragma acc routine seq bind(LAMBDA2)
  auto Lambda2 = [i, F =&j, k, &l](int, float, double){};

  Lambda(1, 2, 3);
  Lambda2(1, 2, 3);
}

// CHECK: cir.func{{.*}} @_Z5Func1v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F1_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F1_R_NAME]] func(@_Z5Func1v) bind("BIND1") seq
//
// CHECK: cir.func{{.*}} @_Z5Func2v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F2_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_Z5Func3v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F3_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F3_R_NAME]] func(@_Z5Func3v) bind("BIND3" [#acc.device_type<nvidia>]) seq
//
// CHECK: cir.func{{.*}} @_Z5Func4v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F4_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_Z5Func5v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F5_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F5_R_NAME]] func(@_Z5Func5v) bind("BIND5_N" [#acc.device_type<nvidia>], "BIND5_N" [#acc.device_type<host>], "BIND5_M" [#acc.device_type<multicore>]) seq
//
// CHECK: cir.func{{.*}} @_Z5Func6v({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F6_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_Z5Func7i({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F7_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F7_R_NAME]] func(@_Z5Func7i) bind(@_Z5BIND7i) seq
//
// CHECK: cir.func{{.*}} @_Z5Func8f({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F8_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_Z5Func9ifs({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F9_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F9_R_NAME]] func(@_Z5Func9ifs) bind(@_Z5BIND9ifs [#acc.device_type<nvidia>]) seq

// CHECK: cir.func{{.*}} @_Z6Func101S({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F10_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_Z6Func11P1UR1Vi({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F11_R_NAME:.*]]]>}
// CHECK: acc.routine @[[F11_R_NAME]] func(@_Z6Func11P1UR1Vi) bind(@_Z10BIND11_NVHP1UR1Vi [#acc.device_type<nvidia>], @_Z10BIND11_NVHP1UR1Vi [#acc.device_type<host>], @_Z9BIND11_MCP1UR1Vi [#acc.device_type<multicore>]) seq
//
// CHECK: cir.func{{.*}} @_Z6Func121U1Vi({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[F12_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_ZN8HasFuncs7MemFuncEidRS_1S({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[MEMFUNC_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_ZNK8HasFuncs12ConstMemFuncEidRS_1S({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[CONSTMEMFUNC_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_ZNVK8HasFuncs15VolatileMemFuncEidRS_1S({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[VOLATILEMEMFUNC_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_ZNKO8HasFuncs10RefMemFuncEidRS_1S({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[REFMEMFUNC_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} @_ZN8HasFuncs13StaticMemFuncEidRS_P1U({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[STATICFUNC_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} lambda{{.*}} @_ZZ10hasLambdasvENK3$_0clEifd({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[LAMBDA1_R_NAME:.*]]]>}
//
// CHECK: cir.func{{.*}} lambda{{.*}} @_ZZ10hasLambdasvENK3$_1clEifd({{.*}}){{.*}} attributes {acc.routine_info = #acc.routine_info<[@[[LAMBDA2_R_NAME:.*]]]>}
//
// CHECK:  acc.routine @[[MEMFUNC_R_NAME]] func(@_ZN8HasFuncs7MemFuncEidRS_1S) bind(@_Z3MEMP8HasFuncsidRS_1S) seq
// CHECK:  acc.routine @[[CONSTMEMFUNC_R_NAME]] func(@_ZNK8HasFuncs12ConstMemFuncEidRS_1S) bind(@_Z3MEMPK8HasFuncsidRS_1S) seq
// CHECK:  acc.routine @[[VOLATILEMEMFUNC_R_NAME]] func(@_ZNVK8HasFuncs15VolatileMemFuncEidRS_1S) bind(@_Z3MEMPVK8HasFuncsidRS_1S) seq
// CHECK:  acc.routine @[[REFMEMFUNC_R_NAME]] func(@_ZNKO8HasFuncs10RefMemFuncEidRS_1S) bind(@_Z3MEMPK8HasFuncsidRS_1S) seq
// CHECK:  acc.routine @[[STATICFUNC_R_NAME]] func(@_ZN8HasFuncs13StaticMemFuncEidRS_P1U) bind(@_Z9STATICMEMP8HasFuncsidRS_P1U) seq
//
// These two LOOK weird because the first argument to each of these is the
// implicit 'this', so they look like they have the lambda mangling (and
// demanglers don't handle lambdas well).
// CHECK:  acc.routine @[[LAMBDA1_R_NAME]] func(@_ZZ10hasLambdasvENK3$_0clEifd) bind(@_Z7LAMBDA1PKZ10hasLambdasvE3$_0ifd) seq
// Manual demangle:
// Func name: _Z7LAMBDA1 -> LAMBDA1
// Args: P -> Pointer
//       K -> Const
//       Z10hasLambdasv-> hasLambdas(void)::
//       E3$_0 -> anonymous type #0
//       ifd -> taking args int, float, double.
// // CHECK:  acc.routine @[[LAMBDA2_R_NAME]] func(@_ZZ10hasLambdasvENK3$_1clEifd) bind(@_Z7LAMBDA2PKZ10hasLambdasvE3$_1ifd) seq

// CHECK: acc.routine @[[F2_R_NAME]] func(@_Z5Func2v) bind("BIND2") seq
// CHECK: acc.routine @[[F4_R_NAME]] func(@_Z5Func4v) bind("BIND4" [#acc.device_type<radeon>]) seq
// CHECK: acc.routine @[[F6_R_NAME]] func(@_Z5Func6v) bind("BIND6_R" [#acc.device_type<radeon>], "BIND6_M" [#acc.device_type<multicore>], "BIND6_M" [#acc.device_type<host>]) seq
// CHECK: acc.routine @[[F8_R_NAME]] func(@_Z5Func8f) bind(@_Z5BIND8f) seq
// CHECK: acc.routine @[[F10_R_NAME]] func(@_Z6Func101S) bind(@_Z6BIND101S [#acc.device_type<radeon>]) seq
// CHECK: acc.routine @[[F12_R_NAME]] func(@_Z6Func121U1Vi) bind(@_Z8BIND12_R1U1Vi [#acc.device_type<radeon>], @_Z10BIND12_MCH1U1Vi [#acc.device_type<multicore>], @_Z10BIND12_MCH1U1Vi [#acc.device_type<host>]) seq

