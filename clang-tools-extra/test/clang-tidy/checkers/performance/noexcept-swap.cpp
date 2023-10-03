// RUN: %check_clang_tidy %s performance-noexcept-swap %t -- -- -fexceptions

void throwing_function() noexcept(false);
void noexcept_function() noexcept;

template <typename>
struct TemplateNoexceptWithInt {
  static void f() {}
};

template <>
struct TemplateNoexceptWithInt<int> {
  static void f() noexcept {}
};

class A {
  void swap(A &);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: swap functions should be marked noexcept [performance-noexcept-swap]
  // CHECK-FIXES: void swap(A &) noexcept ;
};

void swap(A &, A &);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: swap functions should be marked noexcept [performance-noexcept-swap]
// CHECK-FIXES: void swap(A &, A &) noexcept ;

struct B {
  static constexpr bool kFalse = false;
  void swap(B &) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
};

void swap(B &, B &) noexcept(B::kFalse);
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]

template <typename>
struct C {
  void swap(C&);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: swap functions should be marked noexcept [performance-noexcept-swap]
  // CHECK-FIXES: void swap(C&) noexcept ;
};

template <typename T>
void swap(C<T>&, C<T>&);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: swap functions should be marked noexcept [performance-noexcept-swap]
// CHECK-FIXES: void swap(C<T>&, C<T>&) noexcept ;
void swap(C<int>&, C<int>&);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: swap functions should be marked noexcept [performance-noexcept-swap]
// CHECK-FIXES: void swap(C<int>&, C<int>&) noexcept ;

template <typename>
struct D {
  static constexpr bool kFalse = false;
  void swap(D &) noexcept(kFalse);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
};

template <typename T>
void swap(D<T> &, D<T> &) noexcept(D<T>::kFalse);
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
void swap(D<int> &, D<int> &) noexcept(D<int>::kFalse);
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]

struct E {
  void swap(E &) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
};

void swap(E &, E &) noexcept(noexcept(throwing_function()));
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]

template <typename>
struct F {
  void swap(F &) noexcept(noexcept(throwing_function()));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
};

template <typename T>
void swap(F<T> &, F<T> &) noexcept(noexcept(throwing_function()));
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
void swap(F<int> &, F<int> &) noexcept(noexcept(throwing_function()));
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]

struct G {
  void swap(G &) noexcept(noexcept(TemplateNoexceptWithInt<double>::f()));
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]
};

void swap(G &, G &) noexcept(noexcept(TemplateNoexceptWithInt<double>::f()));
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: noexcept specifier on swap function evaluates to 'false' [performance-noexcept-swap]

class OK {};

struct OK1 {
  void swap(OK1 &) noexcept;
};

void swap(OK1 &, OK1 &) noexcept;

struct OK2 {
  static constexpr bool kTrue = true;
  void swap(OK2 &) noexcept(kTrue) {}
};

void swap(OK2 &, OK2 &) noexcept(OK2::kTrue);

struct OK3 {
    void swap(OK3 &) = delete;
};

void swap(OK3 &, OK3 &) = delete;

struct OK4 {
  void swap(OK4 &) noexcept(false);
};

void swap(OK4 &, OK4 &) noexcept(false);

struct OK5 {
  void swap(OK5 &) noexcept(true);
};

void swap(OK5 &, OK5 &)noexcept(true);

struct OK12 {
  void swap(OK12 &) noexcept(noexcept(noexcept_function()));
};

void swap(OK12 &, OK12 &) noexcept(noexcept(noexcept_function()));

struct OK13 {
  void swap(OK13 &) noexcept(noexcept(TemplateNoexceptWithInt<int>::f()));
};

void swap(OK13 &, OK13 &) noexcept(noexcept(TemplateNoexceptWithInt<int>::f()));

template <typename>
class OK14 {};

template <typename>
struct OK15 {
  void swap(OK15 &) noexcept;
};

template <typename T>
void swap(OK15<T> &, OK15<T> &) noexcept;
void swap(OK15<int> &, OK15<int> &) noexcept;

template <typename>
struct OK16 {
  static constexpr bool kTrue = true;
  void swap(OK16 &) noexcept(kTrue);
};

// FIXME: This gives a warning, but it should be OK.
//template <typename T>
//void swap(OK16<T> &, OK16<T> &) noexcept(OK16<T>::kTrue);
template <typename T>
void swap(OK16<int> &, OK16<int> &) noexcept(OK16<int>::kTrue);

template <typename>
struct OK17 {
    void swap(OK17 &) = delete;
};

template <typename T>
void swap(OK17<T> &, OK17<T> &) = delete;
void swap(OK17<int> &, OK17<int> &) = delete;

template <typename>
struct OK18 {
  void swap(OK18 &) noexcept(false);
};

template <typename T>
void swap(OK18<T> &, OK18<T> &) noexcept(false);
void swap(OK18<int> &, OK18<int> &) noexcept(false);

template <typename>
struct OK19 {
  void swap(OK19 &) noexcept(true);
};

template <typename T>
void swap(OK19<T> &, OK19<T> &)noexcept(true);
void swap(OK19<int> &, OK19<int> &)noexcept(true);

template <typename>
struct OK20 {
  void swap(OK20 &) noexcept(noexcept(noexcept_function()));
};

template <typename T>
void swap(OK20<T> &, OK20<T> &) noexcept(noexcept(noexcept_function()));
void swap(OK20<int> &, OK20<int> &) noexcept(noexcept(noexcept_function()));

template <typename>
struct OK21 {
  void swap(OK21 &) noexcept(noexcept(TemplateNoexceptWithInt<int>::f()));
};

template <typename T>
void swap(OK21<T> &, OK21<T> &) noexcept(noexcept(TemplateNoexceptWithInt<int>::f()));
void swap(OK21<int> &, OK21<int> &) noexcept(noexcept(TemplateNoexceptWithInt<int>::f()));

namespace PR64303 {
  void swap();
  void swap(int&, bool&);
  void swap(int&, int&, int&);
  void swap(int&);

  struct Test {
    void swap();
    void swap(Test&, Test&);
    void swap(int&);
    static void swap(int&, int&);

    friend void swap(Test&, Test&);
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: swap functions should be marked noexcept [performance-noexcept-swap]
  };
}
