// RUN: %clang_cc1 -no-opaque-pointers -std=c++98 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=3.0 %s -emit-llvm -o - -Wno-c++11-extensions \
// RUN:     | FileCheck --check-prefixes=CHECK,PRE39,PRE5,PRE12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=3.0 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,PRE39,PRE5,PRE12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=3.8 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,PRE39,PRE5,PRE12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=3.9 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,PRE5,PRE12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=4.0 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,PRE5,PRE12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=5 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,PRE12,PRE12-CXX17 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=11 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,PRE12,PRE12-CXX17 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++20 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=11 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,PRE12,PRE12-CXX17,PRE12-CXX20,PRE13-CXX20 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++20 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=12 %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,V12,V12-CXX17,V12-CXX20,PRE13-CXX20 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++98 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=latest %s -emit-llvm -o - -Wno-c++11-extensions \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,V12 %s
// RUN: %clang_cc1 -no-opaque-pointers -std=c++20 -triple x86_64-linux-gnu -fenable-matrix -fclang-abi-compat=latest %s -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,V39,V5,V12,V12-CXX17,V12-CXX20,V13-CXX20 %s

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

int arg;
template<const int *> struct clang12_unchanged {};
// CHECK: @_Z4test17clang12_unchangedIXadL_Z3argEEE
void test(clang12_unchanged<&arg>) {}

#if __cplusplus >= 201703L
// PRE12-CXX17: @_Z4test15clang12_changedIXadL_Z3argEEE
// V12-CXX17: @_Z4test15clang12_changedIXcvPKiadL_Z3argEEE
template<auto> struct clang12_changed {};
void test(clang12_changed<(const int*)&arg>) {}
#endif

// PRE12: @_Z9clang12_aIXadL_Z3argEEEvv
// V12: @_Z9clang12_aIXcvPKiadL_Z3argEEEvv
template<const int *> void clang12_a() {}
template void clang12_a<&arg>();

// PRE12: @_Z9clang12_bIXadL_Z3arrEEEvv
// V12: @_Z9clang12_bIXadsoKcL_Z3arrEEEEvv
extern const char arr[6] = "hello";
template<const char *> void clang12_b() {}
template void clang12_b<arr>();

// CHECK: @_Z9clang12_cIXadL_Z3arrEEEvv
template<const char (*)[6]> void clang12_c() {}
template void clang12_c<&arr>();


/// Tests for <template-arg> <expr-primary> changes in clang12:
namespace expr_primary {
struct A {
  template<int N> struct Int {};
  template<int& N> struct Ref {};
};

/// Check various DeclRefExpr manglings

// PRE12: @_ZN12expr_primary5test1INS_1AEEEvNT_3IntIXLi1EEEE
// V12:   @_ZN12expr_primary5test1INS_1AEEEvNT_3IntILi1EEE
template <typename T> void test1(typename T::template Int<1> a) {}
template void test1<A>(typename A::template Int<1> a);

enum Enum { EnumVal = 4 };
int Global;

// PRE12: @_ZN12expr_primary5test2INS_1AEEEvNT_3IntIXLNS_4EnumE4EEEE
// V12:   @_ZN12expr_primary5test2INS_1AEEEvNT_3IntILNS_4EnumE4EEE
template <typename T> void test2(typename T::template Int<EnumVal> a) {}
template void test2<A>(typename A::template Int<4> a);

// CHECK: @_ZN12expr_primary5test3ILi3EEEvNS_1A3IntIXT_EEE
template <int X> void test3(typename A::template Int<X> a) {}
template void test3<3>(A::Int<3> a);

#if __cplusplus >= 202002L
// CHECK-CXX20: @_ZN12expr_primary5test4INS_1AEEEvNT_3RefIL_ZNS_6GlobalEEEE
template <typename T> void test4(typename T::template Ref<(Global)> a) {}
template void test4<A>(typename A::template Ref<Global> a);

struct B {
  struct X {
    constexpr X(double) {}
    constexpr X(int&) {}
  };
  template<X> struct Y {};
};

// PRE12-CXX20: _ZN12expr_primary5test5INS_1BEEEvNT_1YIXLd3ff0000000000000EEEE
// V12-CXX20: _ZN12expr_primary5test5INS_1BEEEvNT_1YILd3ff0000000000000EEE
template<typename T> void test5(typename T::template Y<1.0>) { }
template void test5<B>(typename B::Y<1.0>);

// PRE12-CXX20: @_ZN12expr_primary5test6INS_1BEEENT_1YIL_ZZNS_5test6EiE1bEEEi
// V12-CXX20:   @_ZN12expr_primary5test6INS_1BEEENT_1YIXfp_EEEi
template<typename T> auto test6(int b) -> typename T::template Y<b> { return {}; }
template auto test6<B>(int b) -> B::Y<b>;
#endif

/// Verify non-dependent type-traits within a dependent template arg.

// PRE12: @_ZN12expr_primary5test7INS_1AEEEvNT_3IntIXLm1EEEE
// V12:   @_ZN12expr_primary5test7INS_1AEEEvNT_3IntILm1EEE
template <class T> void test7(typename T::template Int<sizeof(char)> a) {}
template void test7<A>(A::Int<1>);

// PRE12: @_ZN12expr_primary5test8ILi2EEEvu11matrix_typeIXLi1EEXT_EiE
// V12:   @_ZN12expr_primary5test8ILi2EEEvu11matrix_typeILi1EXT_EiE
template<int N> using matrix1xN = int __attribute__((matrix_type(1, N)));
template<int N> void test8(matrix1xN<N> a) {}
template void test8<2>(matrix1xN<2> a);

// PRE12: @_ZN12expr_primary5test9EUa9enable_ifIXLi1EEEv
// V12:   @_ZN12expr_primary5test9EUa9enable_ifILi1EEv
void test9(void) __attribute__((enable_if(1, ""))) {}

}

#if __cplusplus >= 202002L
// PRE13-CXX20: @_Z15observe_lambdasI17inline_var_lambdaMUlvE_17inline_var_lambdaMUlvE0_PiS2_S0_S1_EiT_T0_T1_T2_
// V13-CXX20: @_Z15observe_lambdasIN17inline_var_lambdaMUlvE_ENS0_UlvE0_EPiS3_S1_S2_EiT_T0_T1_T2_
template <typename T, typename U, typename V, typename W, typename = T, typename = U>
int observe_lambdas(T, U, V, W) { return 0; }
inline auto inline_var_lambda = observe_lambdas([]{}, []{}, (int*)0, (int*)0);
int use_inline_var_lambda() { return inline_var_lambda; }
#endif
