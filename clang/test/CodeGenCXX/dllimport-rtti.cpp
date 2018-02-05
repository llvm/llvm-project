// RUN: %clang_cc1 -triple i686-windows-msvc -emit-llvm -std=c++1y -fms-extensions -O1 -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple i686-windows-gnu  -emit-llvm -std=c++1y -fms-extensions -O1 -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=GNU

struct __declspec(dllimport) S {
  virtual void f() {}
} s;
// MSVC: [[VF_S:.*]] = private unnamed_addr constant { [2 x i8*] }
// MSVC-DAG: @"\01??_SS@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* [[VF_S]], i32 0, i32 0, i32 1)
// MSVC-DAG: @"\01??_R0?AUS@@@8" = linkonce_odr
// MSVC-DAG: @"\01??_R1A@?0A@EA@S@@8" = linkonce_odr
// MSVC-DAG: @"\01??_R2S@@8" = linkonce_odr
// MSVC-DAG: @"\01??_R3S@@8" = linkonce_odr

// GNU-DAG: @_ZTV1S = available_externally dllimport
// GNU-DAG: @_ZTI1S = linkonce_odr

struct U : S {
} u;

struct __declspec(dllimport) V {
  virtual void f();
} v;
// GNU-DAG: @_ZTV1V = available_externally dllimport
// GNU-DAG: @_ZTS1V = linkonce_odr
// GNU-DAG: @_ZTI1V = linkonce_odr

struct W {
  __declspec(dllimport) virtual void f();
  virtual void g();
} w;
// GNU-DAG: @_ZTV1W = linkonce_odr
// GNU-DAG: @_ZTS1W = linkonce_odr
// GNU-DAG: @_ZTI1W = linkonce_odr
