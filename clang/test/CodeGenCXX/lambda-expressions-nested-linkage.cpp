// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++14 | FileCheck --check-prefixes=CHECK,CXX14 %s

// CHECK-LABEL: define{{.*}} void @_ZN19non_inline_function3fooEv()
// CHECK-LABEL: define internal void @"_ZZN19non_inline_function3fooEvENK3$_0clEi"(%class.anon
// CHECK-LABEL: define internal signext i8 @"_ZZZN19non_inline_function3fooEvENK3$_0clEiENKUlcE_clEc"(%class.anon
namespace non_inline_function {
void foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  L(3)('a');
}
}

namespace non_template {
  struct L {
    int t = ([](int a) { return [](int b) { return b; };})(2)(3);    
  };
  L l; 
}

// It's important that this is not in a namespace; we're testing the mangling
// of lambdas in top-level inline variables here.
inline auto lambda_in_inline_variable = [] {};
template<typename T> struct Wrap {};
// CHECK-LABEL: define {{.*}} @_Z30test_lambda_in_inline_variable4WrapIN25lambda_in_inline_variableMUlvE_EE
void test_lambda_in_inline_variable(Wrap<decltype(lambda_in_inline_variable)>) {}

namespace lambdas_in_NSDMIs_template_class {
template<class T>
struct L {
    T t2 = ([](int a) { return [](int b) { return b; };})(T{})(T{});    
};
L<int> l;
}

// CHECK-LABEL: define linkonce_odr i32 @_ZN15inline_function3fooEv

// CHECK-LABEL: define linkonce_odr void @_ZNK12non_template1L1tMUliE_clEi(%class.anon
// CHECK-LABEL: define linkonce_odr i32 @_ZZNK12non_template1L1tMUliE_clEiENKUliE_clEi(%class.anon


// CHECK-LABEL: define linkonce_odr void @_ZNK32lambdas_in_NSDMIs_template_class1LIiEUliE_clEi(%class.anon
// CHECK-LABEL: define linkonce_odr i32 @_ZZNK32lambdas_in_NSDMIs_template_class1LIiEUliE_clEiENKUliE_clEi(%class.anon

// CHECK-LABEL: define linkonce_odr void @_ZZN15inline_function3fooEvENKUliE_clEi
// CHECK-LABEL: define linkonce_odr signext i8 @_ZZZN15inline_function3fooEvENKUliE_clEiENKUlcE_clEc
namespace inline_function {
inline int foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  L(3)('a');
}
int use = foo();
}

#if __cplusplus >= 201402L
// CXX14-LABEL: define internal void @"_ZZZN32lambda_capture_in_generic_lambda3fooIiEEDavENKUlT_E_clIZNS_L1fEvE3$_1EEDaS1_ENKUlvE_clEv"
namespace lambda_capture_in_generic_lambda {
template <typename T> auto foo() {
  return [](auto func) {
    [func] { func(); }();
  };
}
static void f() {
  foo<int>()([] { });
}
void f1() { f(); }
}
#endif
