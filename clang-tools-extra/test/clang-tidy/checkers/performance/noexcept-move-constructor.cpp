// RUN: %check_clang_tidy %s performance-noexcept-move-constructor %t -- -- -fexceptions

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

struct ThrowingMoveConstructor
{
  ThrowingMoveConstructor() = default;
  ThrowingMoveConstructor(ThrowingMoveConstructor&&) noexcept(false) {
  }
  ThrowingMoveConstructor& operator=(ThrowingMoveConstructor &&) noexcept(false) {
    return *this;
  }
};

class A {
  A(A &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  A &operator=(A &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

struct B {
  static constexpr bool kFalse = false;
  B(B &&) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
};

template <typename>
struct C
{
  C(C &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  C& operator=(C &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

struct D
{
  static constexpr bool kFalse = false;
  D(D &&) noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  D& operator=(D &&) noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false'
};

template <typename>
struct E
{
  static constexpr bool kFalse = false;
  E(E &&) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  E& operator=(E &&) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false'
};

template <typename>
struct F
{
  static constexpr bool kFalse = false;
  F(F &&) noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  F& operator=(F &&) noexcept(kFalse) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false'
};

struct G {
  G(G &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  G& operator=(G &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]

  ThrowingMoveConstructor field;
};

void throwing_function() noexcept(false) {}

struct H {
  H(H &&) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  H &operator=(H &&) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false'
};

template <typename>
struct I {
  I(I &&) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  I &operator=(I &&) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false'
};

template <typename TemplateType> struct TemplatedType {
  static void f() {}
};

template <> struct TemplatedType<int> {
  static void f() noexcept {}
};

struct J {
  J(J &&) noexcept(noexcept(TemplatedType<double>::f()));
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
  J &operator=(J &&) noexcept(noexcept(TemplatedType<double>::f()));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: noexcept specifier on the move assignment operator evaluates to 'false' [performance-noexcept-move-constructor]
};

struct K : public ThrowingMoveConstructor {
  K(K &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  K &operator=(K &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

struct InheritFromThrowingMoveConstrcutor : public ThrowingMoveConstructor
{};

struct L {
  L(L &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  L &operator=(L &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]

  InheritFromThrowingMoveConstrcutor IFF;
};

struct M : public InheritFromThrowingMoveConstrcutor {
  M(M &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  M &operator=(M &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

struct N : public IntWrapper, ThrowingMoveConstructor {
  N(N &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  N &operator=(N &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

struct O : virtual IntWrapper, ThrowingMoveConstructor {
  O(O &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  O &operator=(O &&) = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should be marked noexcept [performance-noexcept-move-constructor]
};

class OK {};

void f() {
  OK a;
  a = OK();
}

class OK1 {
public:
  OK1();
  OK1(const OK1 &);
  OK1(OK1 &&) noexcept;
  OK1 &operator=(OK1 &&) noexcept;
  void f();
  void g() noexcept;
};

struct OK2 {
  static constexpr bool kTrue = true;

  OK2(OK2 &&) noexcept(true) {}
  OK2 &operator=(OK2 &&) noexcept(kTrue) { return *this; }
};

struct OK3 {
  OK3(OK3 &&) noexcept(false) {}
  OK3 &operator=(OK3 &&) = delete;
};

struct OK4 {
  OK4(OK4 &&) noexcept = default;
  OK4 &operator=(OK4 &&) noexcept = default;
};

struct OK5 {
  OK5(OK5 &&) noexcept(true) = default;
  OK5 &operator=(OK5 &&) noexcept(true) = default;
};

struct OK6 {
  OK6(OK6 &&) = default;
  OK6& operator=(OK6 &&) = default;
};

template <typename>
struct OK7 {
  OK7(OK7 &&) = default;
  OK7& operator=(OK7 &&) = default;
};

template <typename>
struct OK8 {
  OK8(OK8 &&) noexcept = default;
  OK8& operator=(OK8 &&) noexcept = default;
};

template <typename>
struct OK9 {
  OK9(OK9 &&) noexcept(true) = default;
  OK9& operator=(OK9 &&) noexcept(true) = default;
};

template <typename>
struct OK10 {
  OK10(OK10 &&) noexcept(false) = default;
  OK10& operator=(OK10 &&) noexcept(false) = default;
};

template <typename>
struct OK11 {
  OK11(OK11 &&) = delete;
  OK11& operator=(OK11 &&) = delete;
};

void noexcept_function() noexcept {}

struct OK12 {
  OK12(OK12 &&) noexcept(noexcept(noexcept_function()));
  OK12 &operator=(OK12 &&) noexcept(noexcept(noexcept_function));
};

struct OK13 {
  OK13(OK13 &&) noexcept(noexcept(noexcept_function)) = default;
  OK13 &operator=(OK13 &&) noexcept(noexcept(noexcept_function)) = default;
};

template <typename> struct OK14 {
  OK14(OK14 &&) noexcept(noexcept(TemplatedType<int>::f()));
  OK14 &operator=(OK14 &&) noexcept(noexcept(TemplatedType<int>::f()));
};

struct OK15 {
  OK15(OK15 &&) = default;
  OK15 &operator=(OK15 &&) = default;

  int member;
};

template <typename>
struct OK16 {
  OK16(OK16 &&) = default;
  OK16 &operator=(OK16 &&) = default;

  int member;
};

struct OK17
{
  OK17(OK17 &&) = default;
  OK17 &operator=(OK17 &&) = default;
  OK empty_field;
};

template <typename>
struct OK18
{
  OK18(OK18 &&) = default;
  OK18 &operator=(OK18 &&) = default;

  OK empty_field;
};

struct OK19 : public OK
{
  OK19(OK19 &&) = default;
  OK19 &operator=(OK19 &&) = default;
};

struct OK20 : virtual OK
{
  OK20(OK20 &&) = default;
  OK20 &operator=(OK20 &&) = default;
};

template <typename T>
struct OK21 : public T
{
  OK21() = default;
  OK21(OK21 &&) = default;
  OK21 &operator=(OK21 &&) = default;
};

template <typename T>
struct OK22 : virtual T
{
  OK22() = default;
  OK22(OK22 &&) = default;
  OK22 &operator=(OK22 &&) = default;
};

template <typename T>
struct OK23 {
  OK23()= default;
  OK23(OK23 &&) = default;
  OK23 &operator=(OK23 &&) = default;

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
  OK24(OK24 &&) = default;
  OK24 &operator=(OK24 &&) = default;
};

struct OK25 : virtual Empty, OK1 {
  OK25(OK25 &&) = default;
  OK25 &operator=(OK25 &&) = default;
};

struct OK26 : public Empty, IntWrapper {
  OK26(OK26 &&) = default;
  OK26 &operator=(OK26 &&) = default;
};

template <typename T>
struct OK27 : public T
{
  OK27(OK27 &&) = default;
  OK27 &operator=(OK27 &&) = default;
};

template <typename T>
struct OK28 : virtual T
{
  OK28(OK28 &&) = default;
  OK28 &operator=(OK28 &&) = default;
};

template <typename T>
struct OK29 {
  OK29(OK29 &&) = default;
  OK29 &operator=(OK29 &&) = default;

  T member;
};

struct OK30 {
  OK30(OK30 &&) noexcept(TrueT<OK30>::value) = default;
  OK30& operator=(OK30 &&) noexcept(TrueT<OK30>::value) = default;
};

template <typename>
struct OK31 {
  OK31(OK31 &&) noexcept(TrueT<int>::value) = default;
  OK31& operator=(OK31 &&) noexcept(TrueT<int>::value) = default;
};
