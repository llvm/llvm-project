// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unused-value -std=c++17
// expected-no-diagnostics
namespace test1 {

template <int num> int return_num() { return num; }

template <typename lambda> struct lambda_wrapper {
  lambda &outer_lambda;
  lambda_wrapper(lambda& outer_lambda) : outer_lambda(outer_lambda) {}
  template <typename T> auto operator+(T t) { outer_lambda(t); return 1; }
};

template <int... nums, typename lambda>
void bw(lambda& outer_lambda) {
  (lambda_wrapper(outer_lambda) + ... + return_num<nums>());
}

template <typename lambda> auto check_return_type(lambda inner_lambda) {
  using inner_lambda_return_type = decltype(inner_lambda(5));
}

void cs() {
  auto outer_lambda = [](auto param) {
    auto inner_lambda = [](auto o) -> decltype(param) {};
    check_return_type(inner_lambda);
  };
  bw<1,2>(outer_lambda);
}

}  // namespace test1

namespace test2 {

template <typename lambda>
auto run_lambda_with_zero(lambda l) {
  l(0);
}
template <typename ... Ts, typename lambda>
void run_lambda_once_per_type(lambda l) {
  ((Ts{}, run_lambda_with_zero(l)), ...);
}
template <typename> void inner_function() {
  char c;
  [](auto param) -> decltype(c) { return param; }(0);
}
void run() {
  auto x = [](auto) -> void { inner_function<int>(); };
  run_lambda_once_per_type<int>(x);
}

}  // namespace test2
