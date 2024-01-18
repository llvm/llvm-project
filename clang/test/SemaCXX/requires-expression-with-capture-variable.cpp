// RUN: %clang -fsyntax-only -std=c++20 -Xclang -verify %s

// expected-no-diagnostics

auto GH69307_Func_1() {
  constexpr auto b = 1;
  return [&](auto c) -> int
           requires requires { b + c; }
  { return 1; };
};
auto GH69307_Func_Ret = GH69307_Func_1()(1);

auto GH69307_Lambda_1 = []() {
  return [&](auto c) -> int
           requires requires { c; }
  { return 1; };
};
auto GH69307_Lambda_1_Ret = GH69307_Lambda_1()(1);

auto GH69307_Lambda_2 = [](auto c) {
  return [&]() -> int
           requires requires { c; }
  { return 1; };
};
auto GH69307_Lambda_2_Ret = GH69307_Lambda_2(1)();
