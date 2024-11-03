// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -emit-obj -debug-info-kind=constructor -std=c++20 %s -o -

namespace PR50787 {
// This code would previously cause a crash.
extern int x_;
consteval auto& X() { return x_; }
constexpr auto& x1 = X();
auto x2 = X();

// CHECK: @_ZN7PR507872x_E = external global i32, align 4
// CHECK-NEXT: @_ZN7PR507872x1E = constant ptr @_ZN7PR507872x_E, align 8
// CHECK-NEXT: @_ZN7PR507872x2E = global i32 0, align 4
}

namespace PR51484 {
// This code would previously cause a crash.
struct X { int val; };
consteval X g() { return {0}; }
void f() { g(); }

// CHECK: define dso_local void @_ZN7PR514841fEv() #1 {
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

namespace Issue55871 {
struct Item {
  consteval Item(char c) :_char{c}{}
  char _char;
};

int function(const Item& item1, const Item& item2) {
  return 0;
}

int foo() {
  return function(Item{'a'}, Item{'a'});
}
} // namespace Issue58871

namespace Issue55065 {
struct Base {
  consteval virtual int Get() const = 0;
};

struct Derived : Base {
  consteval int Get() const override {
    return 42;
  }
};

int foo() {
  constexpr Derived a;

  auto val = a.Get();
  return val;
}
} // namespace Issue55065

namespace GH60166 {

struct Base {
   void* one = nullptr;
   void* two = nullptr;
};

struct Derived : Base {
   void* three = nullptr;
   consteval Derived() = default;
};

void method() {
  // CHECK: %agg.tmp.ensured = alloca %"struct.GH60166::Derived"
  // CHECK: %0 = getelementptr inbounds { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 0
  // CHECK: store ptr null, ptr %0, align 8
  // CHECK: %1 = getelementptr inbounds { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 1
  // CHECK: store ptr null, ptr %1, align 8
  // CHECK: %2 = getelementptr inbounds { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 2
  // CHECK: store ptr null, ptr %2, align 8
   (void)Derived();
}

} // namespace GH60166
