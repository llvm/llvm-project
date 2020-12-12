// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.0 %s -emit-llvm -o - | FileCheck --check-prefixes=PRE39,PRE5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.8 %s -emit-llvm -o - | FileCheck --check-prefixes=PRE39,PRE5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=3.9 %s -emit-llvm -o - | FileCheck --check-prefixes=V39,PRE5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=4.0 %s -emit-llvm -o - | FileCheck --check-prefixes=V39,PRE5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=5 %s -emit-llvm -o - | FileCheck --check-prefixes=V39,V5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=11 %s -emit-llvm -o - | FileCheck --check-prefixes=V11,V5,PRE12 %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=latest %s -emit-llvm -o - | FileCheck --check-prefixes=V39,V5,V12 %s

typedef __attribute__((vector_size(8))) long long v1xi64;
void clang39(v1xi64) {}
// PRE39: @_Z7clang39Dv1_x(i64
// V39: @_Z7clang39Dv1_x(double

struct A {
  A(const A&) = default;
  A(A&&);
};
void clang5(A) {}
// PRE5: @_Z6clang51A()
// V5: @_Z6clang51A(%{{.*}}*

namespace mangle_template_prefix {
  // PRE12: @_ZN22mangle_template_prefix1fINS_1TEEEvNT_1UIiE1VIiEENS4_S5_IfEE(
  // V12: @_ZN22mangle_template_prefix1fINS_1TEEEvNT_1UIiE1VIiEENS5_IfEE(
  template<typename T> void f(typename T::template U<int>::template V<int>, typename T::template U<int>::template V<float>);
  struct T { template<typename I> struct U { template<typename J> using V = int; }; };
  void g() { f<T>(1, 2); }
}

