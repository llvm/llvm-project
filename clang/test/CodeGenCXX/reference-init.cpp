// RUN: %clang_cc1 -triple %itanium_abi_triple -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - -std=c++98 | FileCheck %s --check-prefix=CHECK-CXX98
// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - -std=c++11 | FileCheck %s --check-prefix=CHECK-CXX11
// expected-no-diagnostics

struct XPTParamDescriptor {};
struct nsXPTParamInfo {
  nsXPTParamInfo(const XPTParamDescriptor& desc);
};
void a(XPTParamDescriptor *params) {
  const nsXPTParamInfo& paramInfo = params[0];
}

// CodeGen of reference initialized const arrays.
namespace PR5911 {
  template <typename T, int N> int f(const T (&a)[N]) { return N; }
  int iarr[] = { 1 };
  int test() { return f(iarr); }
}

// radar 7574896
struct Foo { int foo; };
Foo& ignoreSetMutex = *(new Foo);

// Binding to a bit-field that requires a temporary. 
struct { int bitfield : 3; } s = { 3 };
const int &s2 = s.bitfield;

// In C++98, this forms a reference to itself. In C++11 onwards, this performs
// copy-construction.
struct SelfReference { SelfReference &r; };
extern SelfReference self_reference_1;
SelfReference self_reference_2 = {self_reference_1};
// CHECK-CXX98: @self_reference_2 = global %[[SELF_REF:.*]] { %[[SELF_REF]]* @self_reference_1 }
// CHECK-CXX11: @self_reference_2 = global %[[SELF_REF:.*]] zeroinitializer
// CHECK-CXX11: call {{.*}}memcpy{{.*}} @self_reference_2 {{.*}} @self_reference_1
