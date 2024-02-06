// RUN: %clang -fsyntax-only -std=c++20 -Xclang -verify %s

// expected-no-diagnostics

auto GH69307_correct() {
  constexpr auto b = 1;
  // no need to capture
  return [](auto c) -> int
           requires requires { b + c; }
  { return 1; };
};
auto GH69307_correct_ret = GH69307_correct()(1);

auto GH69307_func() {
  constexpr auto b = 1;
  return [&](auto c) -> int
           requires requires { b + c; }
  { return 1; };
};
auto GH69307_func_ret = GH69307_func()(1);

auto GH69307_lambda_1 = []() {
  return [&](auto c) -> int
           requires requires { c; }
  { return 1; };
};
auto GH69307_lambda_1_ret = GH69307_lambda_1()(1);

auto GH69307_lambda_2 = [](auto c) {
  return [&]() -> int
           requires requires { c; }
  { return 1; };
};
auto GH69307_lambda_2_ret = GH69307_lambda_2(1)();
