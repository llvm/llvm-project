// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
#include <typeinfo>

namespace Test1 {

// PR7400
struct A { virtual void f(); };

// CHECK: @_ZN5Test16int_tiE = constant %"class.std::type_info"* bitcast (i8** @_ZTIi to %"class.std::type_info"*), align 8
const std::type_info &int_ti = typeid(int);

// CHECK: @_ZN5Test14A_tiE = constant %"class.std::type_info"* bitcast (i8** @_ZTIN5Test11AE to %"class.std::type_info"*), align 8
const std::type_info &A_ti = typeid(const volatile A &);

volatile char c;

// CHECK: @_ZN5Test14c_tiE = constant %"class.std::type_info"* bitcast (i8** @_ZTIc to %"class.std::type_info"*), align 8
const std::type_info &c_ti = typeid(c);

extern const double &d;

// CHECK: @_ZN5Test14d_tiE = constant %"class.std::type_info"* bitcast (i8** @_ZTId to %"class.std::type_info"*), align 8
const std::type_info &d_ti = typeid(d);

extern A &a;

// CHECK: @_ZN5Test14a_tiE = global
const std::type_info &a_ti = typeid(a);

// CHECK: define i8* @_ZN5Test11fEv
const char *f() {
  try {
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_typeid() noreturn
    return typeid(*static_cast<A *>(0)).name();
  } catch (...) {
    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK-NEXT:   catch i8* null
  }

  return 0;
}

}
