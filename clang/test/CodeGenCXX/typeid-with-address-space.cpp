// RUN: %clang_cc1 -I%S %s -triple amdgcn-amd-amdhsa -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s
// RUN: %clang_cc1 -I%S %s -triple spirv64-unknown-unknown -fsycl-is-device -emit-llvm -fcxx-exceptions -fexceptions -o - | FileCheck %s --check-prefix=WITH-NONZERO-DEFAULT-AS
#include <typeinfo>

namespace Test1 {

// PR7400
struct A { virtual void f(); };

// CHECK: @_ZN5Test16int_tiE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIi to ptr), align 8
// WITH-NONZERO-DEFAULT-AS: @_ZN5Test16int_tiE ={{.*}} constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTIi to ptr addrspace(4)), align 8
const std::type_info &int_ti = typeid(int);

// CHECK: @_ZN5Test14A_tiE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIN5Test11AE to ptr), align 8
// WITH-NONZERO-DEFAULT-AS: @_ZN5Test14A_tiE ={{.*}} constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTIN5Test11AE to ptr addrspace(4)), align 8
const std::type_info &A_ti = typeid(const volatile A &);

volatile char c;

// CHECK: @_ZN5Test14c_tiE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIc to ptr), align 8
// WITH-NONZERO-DEFAULT-AS: @_ZN5Test14c_tiE ={{.*}} constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTIc to ptr addrspace(4)), align 8
const std::type_info &c_ti = typeid(c);

extern const double &d;

// CHECK: @_ZN5Test14d_tiE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTId to ptr), align 8
// WITH-NONZERO-DEFAULT-AS: @_ZN5Test14d_tiE ={{.*}} constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTId to ptr addrspace(4)), align 8
const std::type_info &d_ti = typeid(d);

extern A &a;

// CHECK: @_ZN5Test14a_tiE ={{.*}} global
const std::type_info &a_ti = typeid(a);

// CHECK: @_ZN5Test18A10_c_tiE ={{.*}} constant ptr addrspacecast (ptr addrspace(1) @_ZTIA10_c to ptr), align 8
// WITH-NONZERO-DEFAULT-AS: @_ZN5Test18A10_c_tiE ={{.*}} constant ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZTIA10_c to ptr addrspace(4)), align 8
const std::type_info &A10_c_ti = typeid(char const[10]);

// CHECK-LABEL: define{{.*}} ptr @_ZN5Test11fEv
// CHECK-SAME:  personality ptr @__gxx_personality_v0
// WITH-NONZERO-DEFAULT-AS-LABEL: define{{.*}} ptr addrspace(4) @_ZN5Test11fEv
// WITH-NONZERO-DEFAULT-AS-SAME:  personality ptr @__gxx_personality_v0
const char *f() {
  try {
    // CHECK: br i1
    // CHECK: invoke void @__cxa_bad_typeid() [[NR:#[0-9]+]]
    // WITH-NONZERO-DEFAULT-AS: invoke{{.*}} void @__cxa_bad_typeid() [[NR:#[0-9]+]]
    return typeid(*static_cast<A *>(0)).name();
  } catch (...) {
    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   catch ptr null
    // WITH-NONZERO-DEFAULT-AS:      landingpad { ptr addrspace(4), i32 }
    // WITH-NONZERO-DEFAULT-AS-NEXT:   catch ptr addrspace(4) null
  }

  return 0;
}

}

// CHECK: attributes [[NR]] = { noreturn }
