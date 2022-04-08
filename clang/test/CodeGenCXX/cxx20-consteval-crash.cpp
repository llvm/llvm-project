// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s

namespace PR50787 {
// This code would previously cause a crash.
extern int x_;
consteval auto& X() { return x_; }
constexpr auto& x1 = X();
auto x2 = X();

// CHECK: @_ZN7PR507872x_E = external global i32, align 4
// CHECK-NEXT: @_ZN7PR507872x1E = constant i32* @_ZN7PR507872x_E, align 8
// CHECK-NEXT: @_ZN7PR507872x2E = global i32* @_ZN7PR507872x_E, align 4
}

namespace PR51484 {
// This code would previously cause a crash.
struct X { int val; };
consteval X g() { return {0}; }
void f() { g(); }

// CHECK: define dso_local void @_ZN7PR514841fEv() #0 {
// CHECK: entry:
// CHECK-NOT: call i32 @_ZN7PR514841gEv()
// CHECK:  ret void
// CHECK: }
}

namespace Issue54578 {
inline consteval unsigned char operator""_UC(const unsigned long long n) {
  return static_cast<unsigned char>(n);
}

inline constexpr char f1(const auto octet) {
  return 4_UC;
}

template <typename Ty>
inline constexpr char f2(const Ty octet) {
  return 4_UC;
}

int foo() {
  return f1('a') + f2('a');
}

// Because the consteval functions are inline (implicitly as well as
// explicitly), we need to defer the CHECK lines until this point to get the
// order correct. We want to ensure there is no definition of the consteval
// UDL function, and that the constexpr f1 and f2 functions both return a
// constant value.

// CHECK-NOT: define{{.*}} zeroext i8 @_ZN10Issue54578li3_UCEy
// CHECK: define{{.*}} i32 @_ZN10Issue545783fooEv(
// CHECK: define{{.*}} signext i8 @_ZN10Issue545782f1IcEEcT_(
// CHECK: ret i8 4
// CHECK: define{{.*}} signext i8 @_ZN10Issue545782f2IcEEcT_(
// CHECK: ret i8 4
}
