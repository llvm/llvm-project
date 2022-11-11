// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
#include <typeinfo>

namespace Test1 {

// PR7400
struct A { virtual void f(); };

// CHECK: @_ZN5Test16int_tiE ={{.*}} constant ptr @_ZTIi, align 8
const std::type_info &int_ti = typeid(int);

// CHECK: @_ZN5Test14A_tiE ={{.*}} constant ptr @_ZTIN5Test11AE, align 8
const std::type_info &A_ti = typeid(const volatile A &);

volatile char c;

// CHECK: @_ZN5Test14c_tiE ={{.*}} constant ptr @_ZTIc, align 8
const std::type_info &c_ti = typeid(c);

extern const double &d;

// CHECK: @_ZN5Test14d_tiE ={{.*}} constant ptr @_ZTId, align 8
const std::type_info &d_ti = typeid(d);

extern A &a;

// CHECK: @_ZN5Test14a_tiE ={{.*}} global
const std::type_info &a_ti = typeid(a);

// CHECK: @_ZN5Test18A10_c_tiE ={{.*}} constant ptr @_ZTIA10_c, align 8
const std::type_info &A10_c_ti = typeid(char const[10]);

// CHECK-LABEL: define{{.*}} ptr @_ZN5Test11fEv
// CHECK-SAME:  personality ptr @__gxx_personality_v0
const char *f() {
  try {
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_typeid() [[NR:#[0-9]+]]
    return typeid(*static_cast<A *>(0)).name();
  } catch (...) {
    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   catch ptr null
  }

  return 0;
}

}

// CHECK: attributes [[NR]] = { noreturn }
