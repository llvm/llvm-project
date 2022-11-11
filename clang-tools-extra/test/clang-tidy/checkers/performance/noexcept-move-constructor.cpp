// RUN: %check_clang_tidy %s performance-noexcept-move-constructor %t

class A {
  A(A &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: move constructors should be marked noexcept [performance-noexcept-move-constructor]
  A &operator=(A &&);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: move assignment operators should
};

struct B {
  static constexpr bool kFalse = false;
  B(B &&) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: noexcept specifier on the move constructor evaluates to 'false' [performance-noexcept-move-constructor]
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

class OK2 {
  static constexpr bool kTrue = true;

public:
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
