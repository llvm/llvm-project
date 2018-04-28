// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++1y | FileCheck --check-prefix CHECK_ABI_LATEST %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fblocks -emit-llvm -o - %s -fexceptions -std=c++1y -fclang-abi-compat=6.0 | FileCheck --check-prefix CHECK_ABIV6 %s

// CHECK-LABEL: define void @_ZN19non_inline_function3fooEv
// CHECK-LABEL: define internal void @"_ZZN19non_inline_function3fooEvENK3$_0clEi"(%class.anon
// CHECK-LABEL: define internal signext i8 @"_ZZZN19non_inline_function3fooEvENK3$_0clEiENKUlcE_clEc"(%class.anon
// CHECK-LABEL: define linkonce_odr void @_ZN19non_inline_function4foo2IiEEDav()
namespace non_inline_function {
auto foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  L(3)('a');
  return L;
}

template<typename T> 
auto foo2() {
  return [](const T&) { return 42; };
}

auto use = foo2<int>();

}
//CHECK-LABEL: define linkonce_odr void @_ZN22inline_member_function1X3fooEv(%"struct.inline_member_function::X"* %this)
//CHECK-LABEL: define linkonce_odr void @_ZZN22inline_member_function1X3fooEvENKUliE_clEi(%class.anon
//CHECK-LABEL: define linkonce_odr signext i8 @_ZZZN22inline_member_function1X3fooEvENKUliE_clEiENKUlcE_clEc(%class.anon

namespace inline_member_function {
struct X {
auto foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  return L;
}
};

auto run1 = X{}.foo()(3)('a');

template<typename S>
struct A {
  template<typename T> static auto default_lambda() {
    return [](const T&) { return 42; };
  }

  template<class U = decltype(default_lambda<S>())>
    U func(U u = default_lambda<S>()) { return u; }
  
  template<class T> auto foo() { return [](const T&) { return 42; }; }
};
//CHECK_ABIV6: define linkonce_odr i32 @_ZZN22inline_member_function1AIdE14default_lambdaIdEEDavENKUlRKdE_clES5_(%class.anon
//CHECK_ABI_LATEST: define linkonce_odr i32 @_ZZN22inline_member_function1AIdE14default_lambdaIdEEDavENKUlRKdE_clES4_(%class.anon
int run2 = A<double>{}.func()(3.14);

//CHECK_ABIV6: define linkonce_odr i32 @_ZZN22inline_member_function1AIcE14default_lambdaIcEEDavENKUlRKcE_clES5_(%class.anon
//CHECK_ABI_LATEST: define linkonce_odr i32 @_ZZN22inline_member_function1AIcE14default_lambdaIcEEDavENKUlRKcE_clES4_(%class.anon
int run3 = A<char>{}.func()('a');
} // end inline_member_function


// CHECK-LABEL: define linkonce_odr void @_ZN15inline_function3fooEv()
// CHECK: define linkonce_odr void @_ZZN15inline_function3fooEvENKUliE_clEi(%class.anon
// CHECK: define linkonce_odr signext i8 @_ZZZN15inline_function3fooEvENKUliE_clEiENKUlcE_clEc(%class.anon
namespace inline_function {
inline auto foo() {
  auto L = [](int a) {
    return [](char b) {
     return b;
    };
  };
  return L;
}
auto use = foo()(3)('a');
}

