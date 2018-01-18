// RUN: %check_clang_tidy %s fuchsia-trailing-return %t

int add_one(const int arg) { return arg; }

auto get_add_one() -> int (*)(const int) {
  // CHECK-MESSAGES: [[@LINE-1]]:1: warning: a trailing return type is disallowed for this type of declaration
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
