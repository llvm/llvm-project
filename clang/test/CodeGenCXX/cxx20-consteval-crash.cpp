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
  // CHECK: %0 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 0
  // CHECK: store ptr null, ptr %0, align 8
  // CHECK: %1 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 1
  // CHECK: store ptr null, ptr %1, align 8
  // CHECK: %2 = getelementptr inbounds nuw { ptr, ptr, ptr }, ptr %agg.tmp.ensured, i32 0, i32 2
  // CHECK: store ptr null, ptr %2, align 8
   (void)Derived();
}

} // namespace GH60166

namespace GH61142 {

template <typename T>
struct Test {
  constexpr static void bar() {
    foo();
  }
  consteval static void foo() {};
};

consteval void a() {
  Test<int>::bar();
}

void b() {
  Test<int>::bar();
}

// Make sure consteval function is not emitted.
// CHECK-NOT: call {{.*}}foo{{.*}}()
// CHECK-NOT: define {{.*}}foo{{.*}}()

} // namespace GH61142

namespace GH219272 {

consteval void f() {}
void g();

struct S {
  consteval S() { f(); }
  consteval S(int) { f(); }
  consteval S(int, int) { f(); }
};

struct D {
  consteval D(int) { f(); }
  constexpr ~D() {
    if (!__builtin_is_constant_evaluated())
      g();
  }
};

template <typename T> void dtor(T) { (void)D{1}; }
template <typename T> void dtor2(T) { (void)D{1}; (void)D{2}; }
template <typename T> void braces(T) { (void)S{1}; }
template <typename T> void parens(T) { (void)S(1, 2); }
template <typename T> void empty_braces(T) { (void)S{}; }
template <typename T> void lambda(T) { [](auto) { (void)S{1}; }(0); }
template <typename T> struct C {
  void m() { (void)S{1}; }
};
template <typename T> void member(T) { C<T>{}.m(); }

struct M {
  consteval int m() const { f(); return 1; }
};
constexpr M gm{};
template <typename T> int memcall(T) { return gm.m(); }

template int memcall<int>(int);
template void dtor<int>(int);
template void dtor2<int>(int);
template void braces<int>(int);
template void parens<int>(int);
template void empty_braces<int>(int);
template void lambda<int>(int);
template void member<int>(int);

// A consteval member call on a non-dependent object is reused as well.
// CHECK: define {{.*}} @_ZN8GH2192727memcallIiEEiT_(
// CHECK-NOT: call
// CHECK: ret i32 1

// The temporary is constant-evaluated, but its destructor still runs.
// CHECK: define {{.*}} @_ZN8GH2192724dtorIiEEvT_(
// CHECK-NOT: call {{.*}}GH2192721DC
// CHECK: call void @_ZN8GH2192721DD1Ev(

// Same with two immediate invocations in one body (Sema rewrites them then).
// CHECK: define {{.*}} @_ZN8GH2192725dtor2IiEEvT_(
// CHECK-NOT: call {{.*}}GH2192721DC
// CHECK: call void @_ZN8GH2192721DD1Ev(
// CHECK-NOT: call {{.*}}GH2192721DC
// CHECK: call void @_ZN8GH2192721DD1Ev(

// Make sure the consteval constructors are neither called nor emitted.
// CHECK-NOT: call {{.*}}GH2192721{{S|D}}C
// CHECK-NOT: define {{.*}}GH2192721{{S|D}}C
// CHECK-NOT: define {{.*}}GH2192721M1m

} // namespace GH219272
