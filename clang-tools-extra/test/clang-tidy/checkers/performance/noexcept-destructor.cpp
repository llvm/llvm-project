// RUN: %check_clang_tidy %s performance-noexcept-destructor %t -- -- -fexceptions

struct Empty
{};

struct IntWrapper {
  int value;
};

template <typename T>
struct FalseT {
  static constexpr bool value = false;
};

template <typename T>
struct TrueT {
  static constexpr bool value = true;
};

struct ThrowOnAnything {
  ThrowOnAnything() noexcept(false);
  ThrowOnAnything(ThrowOnAnything&&) noexcept(false);
  ThrowOnAnything& operator=(ThrowOnAnything &&) noexcept(false);
  ~ThrowOnAnything() noexcept(false);
};

struct B {
  static constexpr bool kFalse = false;
  ~B() noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

struct D {
  static constexpr bool kFalse = false;
  ~D() noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

template <typename>
struct E {
  static constexpr bool kFalse = false;
  ~E() noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false'
};

template <typename>
struct F {
  static constexpr bool kFalse = false;
  ~F() noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

struct G {
  ~G() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~G() noexcept  = default;

  ThrowOnAnything field;
};

void throwing_function() noexcept(false) {}

struct H {
  ~H() noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

template <typename>
struct I {
  ~I() noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

template <typename T> struct TemplatedType {
  static void f() {}
};

template <> struct TemplatedType<int> {
  static void f() noexcept {}
};

struct J {
  ~J() noexcept(noexcept(TemplatedType<double>::f()));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: noexcept specifier on the destructor evaluates to 'false' [performance-noexcept-destructor]
};

struct K : public ThrowOnAnything {
  ~K() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~K() noexcept  = default;
};

struct InheritFromThrowOnAnything : public ThrowOnAnything
{};

struct L {
  ~L() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~L() noexcept  = default;

  InheritFromThrowOnAnything IFF;
};

struct M : public InheritFromThrowOnAnything {
  ~M() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~M() noexcept  = default;
};

struct N : public IntWrapper, ThrowOnAnything {
  ~N() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~N() noexcept  = default;
};

struct O : virtual IntWrapper, ThrowOnAnything {
  ~O() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: destructors should be marked noexcept [performance-noexcept-destructor]
  // CHECK-FIXES: ~O() noexcept  = default;
};

class OK {};

struct OK1 {
  ~OK1() noexcept;
};

struct OK2 {
  static constexpr bool kTrue = true;

  ~OK2() noexcept(true) {}
};

struct OK4 {
  ~OK4() noexcept(false) {}
};

struct OK3 {
  ~OK3() = default;
};

struct OK5 {
  ~OK5() noexcept(true) = default;
};

struct OK6 {
  ~OK6() = default;
};

template <typename>
struct OK7 {
  ~OK7() = default;
};

template <typename>
struct OK8 {
  ~OK8() noexcept = default;
};

template <typename>
struct OK9 {
  ~OK9() noexcept(true) = default;
};

template <typename>
struct OK10 {
  ~OK10() noexcept(false) = default;
};

template <typename>
struct OK11 {
  ~OK11() = delete;
};

void noexcept_function() noexcept {}

struct OK12 {
  ~OK12() noexcept(noexcept(noexcept_function()));
};

struct OK13 {
  ~OK13() noexcept(noexcept(noexcept_function())) = default;
};

template <typename>
struct OK14 {
  ~OK14() noexcept(noexcept(TemplatedType<int>::f()));
};

struct OK15 {
  ~OK15() = default;

  int member;
};

template <typename>
struct OK16 {
  ~OK16() = default;

  int member;
};

struct OK17 {
  ~OK17() = default;

  OK empty_field;
};

template <typename>
struct OK18 {
  ~OK18() = default;

  OK empty_field;
};

struct OK19 : public OK {
  ~OK19() = default;
};

struct OK20 : virtual OK {
  ~OK20() = default;
};

template <typename T>
struct OK21 : public T {
  ~OK21() = default;
};

template <typename T>
struct OK22 : virtual T {
  ~OK22() = default;
};

template <typename T>
struct OK23 {
  ~OK23() = default;

  T member;
};

void testTemplates() {
  OK21<Empty> value(OK21<Empty>{});
  value = OK21<Empty>{};

  OK22<Empty> value2{OK22<Empty>{}};
  value2 = OK22<Empty>{};

  OK23<Empty> value3{OK23<Empty>{}};
  value3 =OK23<Empty>{};
}

struct OK24 : public Empty, OK1 {
  ~OK24() = default;
};

struct OK25 : virtual Empty, OK1 {
  ~OK25() = default;
};

struct OK26 : public Empty, IntWrapper {
  ~OK26() = default;
};

template <typename T>
struct OK27 : public T {
  ~OK27() = default;
};

template <typename T>
struct OK28 : virtual T {
  ~OK28() = default;
};

template <typename T>
struct OK29 {
  ~OK29() = default;

  T member;
};

struct OK30 {
  ~OK30() noexcept(TrueT<OK30>::value) = default;
};

template <typename>
struct OK31 {
  ~OK31() noexcept(TrueT<int>::value) = default;
};

struct OK32 {
  ~OK32();
};

template <typename>
struct OK33 {
  ~OK33();
};

struct OK34 {
  ~OK34() {}
};

template <typename>
struct OK35 {
  ~OK35() {}
};
