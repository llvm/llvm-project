// RUN: %check_clang_tidy %s modernize-pass-by-value %t -- -- -fno-delayed-template-parsing

namespace std {
template <typename T>
struct remove_reference { typedef T type; };
template <typename T>
struct remove_reference<T &> { typedef T type; };
template <typename T>
struct remove_reference<T &&> { typedef T type; };

template <typename T>
typename remove_reference<T>::type &&move(T &&) noexcept;
} // namespace std

struct Movable {
  int a, b, c;
  Movable() = default;
  Movable(const Movable &) {}
  Movable(Movable &&) {}
};

struct NotMovable {
  NotMovable() = default;
  NotMovable(const NotMovable &) = default;
  NotMovable(NotMovable &&) = delete;
  int a, b, c;
};

// POD types are trivially move constructible.
struct POD {
  int a, b, c;
};

// Positive: const string-like ref copied into local variable in free function.
void take_movable(const Movable &M) {
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: pass by value and use std::move [modernize-pass-by-value]
  // CHECK-FIXES: void take_movable(Movable M) {
  Movable Local = M;
  // CHECK-FIXES: Movable Local = std::move(M);
  (void)Local;
}

// Positive: const ref copied in a method.
struct MyClass {
  void process(const Movable &M) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pass by value and use std::move [modernize-pass-by-value]
    // CHECK-FIXES: void process(Movable M) {
    Movable Copy = M;
    // CHECK-FIXES: Movable Copy = std::move(M);
    (void)Copy;
  }
};

// Positive: separate declaration and definition.
void with_decl(const Movable &M);
// CHECK-FIXES: void with_decl(Movable M);
void with_decl(const Movable &M) {
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pass by value and use std::move [modernize-pass-by-value]
  // CHECK-FIXES: void with_decl(Movable M) {
  Movable Local = M;
  // CHECK-FIXES: Movable Local = std::move(M);
  (void)Local;
}

// Negative: trivially copyable type.
void take_pod(const POD &P) {
  POD Local = P;
  (void)Local;
}

// Negative: no copy, just using the reference.
void use_ref(const Movable &M) {
  (void)M.a;
}

// Negative: type has no move constructor (deleted).
void take_nomove(const NotMovable &N) {
  NotMovable Local = N;
  (void)Local;
}

// Negative: parameter used more than once.
void multi_use(const Movable &M) {
  Movable Local = M;
  (void)Local;
  (void)M.a;
}

// Negative: parameter is not const ref.
void take_value(Movable M) {
  Movable Local = M;
  (void)Local;
}

// Negative: not a parameter, just a local.
void local_copy() {
  Movable A;
  Movable B = A;
  (void)B;
}

// Negative: rvalue overload exists.
void with_rvalue_overload(const Movable &M) {
  Movable Local = M;
  (void)Local;
}
void with_rvalue_overload(Movable &&M);

// Negative: template function (dependent context).
template <typename T>
void take_template(const T &M) {
  T Local = M;
  (void)Local;
}
