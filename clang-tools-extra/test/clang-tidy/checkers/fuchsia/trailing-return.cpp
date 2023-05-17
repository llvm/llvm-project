// RUN: %check_clang_tidy -std=c++17-or-later %s fuchsia-trailing-return %t

int add_one(const int arg) { return arg; }

auto get_add_one() -> int (*)(const int) {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: a trailing return type is disallowed for this function declaration
  // CHECK-NEXT: auto get_add_one() -> int (*)(const int) {
  return add_one;
}

auto lambda = [](double x, double y) {return x + y;};

auto lambda2 = [](double x, double y) -> double {return x + y;};

int main() {
  get_add_one()(5);
  return 0;
}

template <typename T1, typename T2>
auto fn(const T1 &lhs, const T2 &rhs) -> decltype(lhs + rhs) {
  return lhs + rhs;
}

// Now check that implicit and explicit C++17 deduction guides don't trigger this warning (PR#47614).

template <typename T>
struct ImplicitDeductionGuides {
  ImplicitDeductionGuides(const T &);
};

template <typename A, typename B>
struct pair {
  A first;
  B second;
};

template <typename T>
struct UserDefinedDeductionGuides {
  UserDefinedDeductionGuides(T);
  template <typename T1, typename T2>
  UserDefinedDeductionGuides(T1, T2);
};

template <typename T1, typename T2>
UserDefinedDeductionGuides(T1, T2) -> UserDefinedDeductionGuides<pair<T1, T2>>;

void foo() {
  ImplicitDeductionGuides X(42);
  UserDefinedDeductionGuides s(1, "abc");
}
