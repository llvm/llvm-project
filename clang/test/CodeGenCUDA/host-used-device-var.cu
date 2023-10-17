// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip %s \
// RUN:   -std=c++17 -O3 -emit-llvm -o - | FileCheck -check-prefix=HOST %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++17 -O3 -mllvm -amdgpu-internalize-symbols -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=DEV-NEG %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip %s \
// RUN:   -std=c++17 -O3 -emit-llvm -o - | FileCheck -check-prefix=HOST-NEG %s

#include "Inputs/cuda.h"

// DEV-DAG: @v1
__device__ int v1;

// DEV-DAG: @v2
__constant__ int v2;

// Check device variables used by neither host nor device functioins are not kept.

// DEV-NEG-NOT: @_ZL2v3
static __device__ int v3;

// Check device variables used by host functions are kept.

// DEV-DAG: @u1
__device__ int u1;

// DEV-DAG: @u2
__constant__ int u2;

// Check host-used static device var is in llvm.compiler.used.
// DEV-DAG: @_ZL2u3
static __device__ int u3;

// Check device-used static device var is emitted but is not in llvm.compiler.used.
// DEV-DAG: @_ZL2u4
static __device__ int u4;

// Check device variables with used attribute are always kept.
// DEV-DAG: @u5
__device__ __attribute__((used)) int u5;

// Test external device variable ODR-used by host code is not emitted or registered.
// DEV-NEG-NOT: @ext_var
extern __device__ int ext_var;

// DEV-DAG: @inline_var = linkonce_odr addrspace(1) externally_initialized global i32 0
__device__ inline int inline_var;

template<typename T>
using func_t = T (*) (T, T);

template <typename T>
__device__ T add_func (T x, T y)
{
  return x + y;
}

// DEV-DAG: @_Z10p_add_funcIiE = linkonce_odr addrspace(1) externally_initialized global ptr @_Z8add_funcIiET_S0_S0_
template <typename T>
__device__ func_t<T> p_add_func = add_func<T>;

// Check non-constant constexpr variables ODR-used by host code only is not emitted.
// DEV-NEG-NOT: constexpr_var1a
// DEV-NEG-NOT: constexpr_var1b
constexpr int constexpr_var1a = 1;
inline constexpr int constexpr_var1b = 1;

// Check constant constexpr variables ODR-used by host code only.
// Device-side constexpr variables accessed by host code should be externalized and kept.
// DEV-DAG: @_ZL15constexpr_var2a = addrspace(4) externally_initialized constant i32 2
// DEV-DAG: @constexpr_var2b = linkonce_odr addrspace(4) externally_initialized constant i32 2
__constant__ constexpr int constexpr_var2a = 2;
inline __constant__ constexpr int constexpr_var2b = 2;

void use(func_t<int> p);
__host__ __device__ void use(const int *p);

// Check static device variable in host function.
// DEV-DAG:  @_ZZ4fun1vE11static_var1 = addrspace(1) externally_initialized global i32 3
void fun1() {
  static __device__ int static_var1 = 3;
  use(&u1);
  use(&u2);
  use(&u3);
  use(&ext_var);
  use(&inline_var);
  use(p_add_func<int>);
  use(&constexpr_var1a);
  use(&constexpr_var1b);
  use(&constexpr_var2a);
  use(&constexpr_var2b);
  use(&static_var1);
}

// Check static variable in host device function.
// DEV-DAG:  @_ZZ4fun2vE11static_var2 = internal addrspace(1) global i32 4
// DEV-DAG:  @_ZZ4fun2vE11static_var3 = addrspace(1) global i32 4
__host__ __device__ void fun2() {
  static int static_var2 = 4;
  static __device__ int static_var3 = 4;
  use(&static_var2);
  use(&static_var3);
}

__global__ void kern1(int **x) {
  *x = &u4;
  fun2();
}

// Check static variables of lambda functions.

// Lambda functions are implicit host device functions.
// Default static variables in lambda functions should be treated
// as host variables on host side, therefore should not be forced
// to be emitted on device.

// DEV-DAG: @_ZZZN21TestStaticVarInLambda3funEvENKUlPcE_clES0_E4var2 = addrspace(1) externally_initialized global i32 5
// DEV-NEG-NOT: @_ZZZN21TestStaticVarInLambda3funEvENKUlPcE_clES0_E4var1
namespace TestStaticVarInLambda {
class A {
public:
  A(char *);
};
void fun() {
  (void) [](char *c) {
    static A var1(c);
    static __device__ int var2 = 5;
    (void) var1;
    (void) var2;
  };
}
}

// Check implicit constant variable ODR-used by host code is not emitted.

// AST contains instantiation of al<ar>, which triggers AST instantiation
// of x::al<ar>::am, which triggers AST instatiation of x::ap<ar>,
// which triggers AST instantiation of aw<ar>::c, which has type
// ar. ar has base class x which has member ah. x::ah is initialized
// with function pointer pointing to ar:as, which returns an object
// of type ou. The constexpr aw<ar>::c is an implicit constant variable
// which is ODR-used by host function x::ap<ar>. An incorrect implementation
// will force aw<ar>::c to be emitted on device side, which will trigger
// emit of x::as and further more ctor of ou and variable o.
// The ODR-use of aw<ar>::c in x::ap<ar> should be treated as a host variable
// instead of device variable.

// DEV-NEG-NOT: _ZN16TestConstexprVar1oE
namespace TestConstexprVar {
char o;
class ou {
public:
  ou(char) { __builtin_strlen(&o); }
};
template < typename ao > struct aw { static constexpr ao c; };
class x {
protected:
  typedef ou (*y)(const x *);
  constexpr x(y ag) : ah(ag) {}
  template < bool * > struct ak;
  template < typename > struct al {
    static bool am;
    static ak< &am > an;
  };
  template < typename ao > static x ap() { (void)aw< ao >::c; return x(nullptr); }
  y ah;
};
template < typename ao > bool x::al< ao >::am(&ap< ao >);
class ar : x {
public:
  constexpr ar() : x(as) {}
  static ou as(const x *) { return 0; }
  al< ar > av;
};
}

// Check the exact list of variables to ensure @_ZL2u4 is not among them.
// DEV: @llvm.compiler.used = {{[^@]*}} @_Z10p_add_funcIiE
// DEV-SAME: {{^[^@]*}} @_ZL15constexpr_var2a
// DEV-SAME: {{^[^@]*}} @_ZL2u3
// DEV-SAME: {{^[^@]*}} @_ZZ4fun1vE11static_var1
// DEV-SAME: {{^[^@]*}} @_ZZZN21TestStaticVarInLambda3funEvENKUlPcE_clES0_E4var2
// DEV-SAME: {{^[^@]*}} @constexpr_var2b
// DEV-SAME: {{^[^@]*}} @inline_var
// DEV-SAME: {{^[^@]*}} @u1
// DEV-SAME: {{^[^@]*}} @u2
// DEV-SAME: {{^[^@]*}} @u5
// DEV-SAME: {{^[^@]*$}}

// HOST-DAG: hipRegisterVar{{.*}}@u1
// HOST-DAG: hipRegisterVar{{.*}}@u2
// HOST-DAG: hipRegisterVar{{.*}}@_ZL2u3
// HOST-DAG: hipRegisterVar{{.*}}@constexpr_var2b
// HOST-DAG: hipRegisterVar{{.*}}@u5
// HOST-DAG: hipRegisterVar{{.*}}@inline_var
// HOST-DAG: hipRegisterVar{{.*}}@_Z10p_add_funcIiE
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZZ4fun1vE11static_var1
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZZ4fun2vE11static_var2
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZZ4fun2vE11static_var3
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZZZN21TestStaticVarInLambda3funEvENKUlPcE_clES0_E4var2
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZZZN21TestStaticVarInLambda3funEvENKUlPcE_clES0_E4var1
// HOST-NEG-NOT: hipRegisterVar{{.*}}@ext_var
// HOST-NEG-NOT: hipRegisterVar{{.*}}@_ZL2u4
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var1a
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var1b
// HOST-NEG-NOT: hipRegisterVar{{.*}}@constexpr_var2a
