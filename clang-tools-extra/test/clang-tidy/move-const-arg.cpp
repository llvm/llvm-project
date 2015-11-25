// RUN: %check_clang_tidy %s misc-move-const-arg %t -- -- -std=c++11

namespace std {
template <typename> struct remove_reference;

template <typename _Tp> struct remove_reference { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t);

} // namespace std

class A {
public:
  A() {}
  A(const A &rhs) {}
  A(A &&rhs) {}
};

int f1() {
  return std::move(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the expression of trivially-copyable type has no effect; remove std::move() [misc-move-const-arg]
  // CHECK-FIXES: return 42;
}

int f2(int x2) {
  return std::move(x2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable of trivially-copyable type
  // CHECK-FIXES: return x2;
}

int *f3(int *x3) {
  return std::move(x3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable of trivially-copyable type
  // CHECK-FIXES: return x3;
}

A f4(A x4) { return std::move(x4); }

A f5(const A x5) {
  return std::move(x5);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable
  // CHECK-FIXES: return x5;
}

template <typename T> T f6(const T x6) { return std::move(x6); }

void f7() { int a = f6(10); }

#define M1(x) x
void f8() {
  const A a;
  M1(A b = std::move(a);)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: std::move of the const variable
  // CHECK-FIXES: M1(A b = a;)
}

#define M2(x) std::move(x)
int f9() { return M2(1); }

template <typename T> T f10(const int x10) {
  return std::move(x10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable
  // CHECK-FIXES: return x10;
}
void f11() {
  f10<int>(1);
  f10<double>(1);
}
