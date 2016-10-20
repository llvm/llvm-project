// RUN: %check_clang_tidy %s modernize-avoid-bind %t -- -- -std=c++14

namespace std {
inline namespace impl {
template <class Fp, class... Arguments>
class bind_rt {};

template <class Fp, class... Arguments>
bind_rt<Fp, Arguments...> bind(Fp &&, Arguments &&...);
}
}

int add(int x, int y) { return x + y; }

void f() {
  auto clj = std::bind(add, 2, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind [modernize-avoid-bind]
  // CHECK-FIXES: auto clj = [] { return add(2, 2); };
}

void g() {
  int x = 2;
  int y = 2;
  auto clj = std::bind(add, x, y);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto clj = [=] { return add(x, y); };
}

struct placeholder {};
placeholder _1;
placeholder _2;

void h() {
  int x = 2;
  auto clj = std::bind(add, x, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto clj = [=](auto && arg1) { return add(x, arg1); };
}

struct A;
struct B;
bool ABTest(const A &, const B &);

void i() {
  auto BATest = std::bind(ABTest, _2, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto BATest = [](auto && arg1, auto && arg2) { return ABTest(arg2, arg1); };
}

void j() {
  auto clj = std::bind(add, 2, 2, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // No fix is applied for argument mismatches.
  // CHECK-FIXES: auto clj = std::bind(add, 2, 2, 2);
}

void k() {
  auto clj = std::bind(add, _1, _1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // No fix is applied for reused placeholders.
  // CHECK-FIXES: auto clj = std::bind(add, _1, _1);
}

void m() {
  auto clj = std::bind(add, 1, add(2, 5));
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // No fix is applied for nested calls.
  // CHECK-FIXES: auto clj = std::bind(add, 1, add(2, 5));
}

namespace C {
  int add(int x, int y){ return x + y; }
}

void n() {
  auto clj = std::bind(C::add, 1, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: prefer a lambda to std::bind
  // CHECK-FIXES: auto clj = [] { return C::add(1, 1); };
}
