// RUN: %check_clang_tidy %s misc-move-forwarding-reference %t -- -- -std=c++14 -fno-delayed-template-parsing

namespace std {
template <typename> struct remove_reference;

template <typename _Tp> struct remove_reference { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t);

} // namespace std

// Standard case.
template <typename T, typename U> void f1(U &&SomeU) {
  T SomeT(std::move(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
  // CHECK-FIXES: T SomeT(std::forward<U>(SomeU));
}

// Ignore parentheses around the argument to std::move().
template <typename T, typename U> void f2(U &&SomeU) {
  T SomeT(std::move((SomeU)));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
  // CHECK-FIXES: T SomeT(std::forward<U>((SomeU)));
}

// Handle the case correctly where std::move() is being used through a using
// declaration.
template <typename T, typename U> void f3(U &&SomeU) {
  using std::move;
  T SomeT(move(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
  // CHECK-FIXES: T SomeT(std::forward<U>(SomeU));
}

// Handle the case correctly where a global specifier is prepended to
// std::move().
template <typename T, typename U> void f4(U &&SomeU) {
  T SomeT(::std::move(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
  // CHECK-FIXES: T SomeT(::std::forward<U>(SomeU));
}

// Create a correct fix if there are spaces around the scope resolution
// operator.
template <typename T, typename U> void f5(U &&SomeU) {
  {
    T SomeT(::  std  ::  move(SomeU));
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: forwarding reference passed to
    // CHECK-FIXES: T SomeT(::std::forward<U>(SomeU));
  }
  {
    T SomeT(std  ::  move(SomeU));
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: forwarding reference passed to
    // CHECK-FIXES: T SomeT(std::forward<U>(SomeU));
  }
}

// Ignore const rvalue reference parameters.
template <typename T, typename U> void f6(const U &&SomeU) {
  T SomeT(std::move(SomeU));
}

// Ignore the case where the argument to std::move() is a lambda parameter (and
// thus not actually a parameter of the function template).
template <typename T, typename U> void f7() {
  [](U &&SomeU) { T SomeT(std::move(SomeU)); };
}

// Ignore the case where the argument is a lvalue reference.
template <typename T, typename U> void f8(U &SomeU) {
  T SomeT(std::move(SomeU));
}

// Ignore the case where the template parameter is a class template parameter
// (i.e. no template argument deduction is taking place).
template <typename T, typename U> class SomeClass {
  void f(U &&SomeU) { T SomeT(std::move(SomeU)); }
};

// Ignore the case where the function parameter in the template isn't an rvalue
// reference but the template argument is explicitly set to be an rvalue
// reference.
class A {};
template <typename T> void foo(T);
void f8() {
  A a;
  foo<A &&>(std::move(a));
}

// A warning is output, but no fix is suggested, if a macro is used to rename
// std::move.
#define MOVE(x) std::move((x))
template <typename T, typename U> void f9(U &&SomeU) {
  T SomeT(MOVE(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
}

// Same result if the argument is passed outside of the macro.
#undef MOVE
#define MOVE std::move
template <typename T, typename U> void f10(U &&SomeU) {
  T SomeT(MOVE(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
}

// Same result if the macro does not include the "std" namespace.
#undef MOVE
#define MOVE move
template <typename T, typename U> void f11(U &&SomeU) {
  T SomeT(std::MOVE(SomeU));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: forwarding reference passed to
}

// Handle the case correctly where the forwarding reference is a parameter of a
// generic lambda.
template <typename T> void f12() {
  [] (auto&& x) { T SomeT(std::move(x)); };
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: forwarding reference passed to
  // CHECK-FIXES: [] (auto&& x) { T SomeT(std::forward<decltype(x)>(x)); }
}
