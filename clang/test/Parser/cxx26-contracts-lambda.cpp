// RUN: %clang_cc1 -std=c++2c -fcontracts -fsyntax-only -verify %s

template <typename>
concept LambdaConstraint = true;

void lambda_contracts() {
  auto no_parentheses = [] pre(true) { return 0; };
  auto explicit_return = [](int value) -> int
      pre(value > 0) post(result: result >= value) { return value; };
  auto qualified = [](int value) mutable noexcept -> int
      pre(value > 0) post(value >= 0) { return value; };
  auto constexpr_lambda = [](int value) constexpr pre(value > 0) {
    return value;
  };
  auto static_lambda = [](int value) static noexcept pre(value > 0) {
    return value;
  };
  auto with_default = [](int value = 1) pre(value > 0) { return value; };

  auto generic = []<typename T>(T value) pre(value > T{}) { return value; };
  auto constrained = []<typename T>(T value)
      requires LambdaConstraint<T>
      pre(value > T{}) post(value >= T{}) { return value; };
  auto doubly_constrained = []<typename T>
      requires LambdaConstraint<T>
      (T value) -> T requires LambdaConstraint<T>
      pre(value > T{}) post(value >= T{}) { return value; };
  auto non_type_parameter = []<int N>() pre(N > 0) { return N; };
  auto parameter_pack = []<typename... T>(T... values)
      pre(sizeof...(values) > 0) { return sizeof...(values); };

  auto nested = [](int outer) pre(outer > 0) {
    return [outer](int inner) pre(inner > outer) { return inner; };
  };

  (void)no_parentheses;
  (void)explicit_return;
  (void)qualified;
  (void)constexpr_lambda;
  (void)static_lambda;
  (void)with_default;
  (void)generic;
  (void)constrained;
  (void)doubly_constrained;
  (void)non_type_parameter;
  (void)parameter_pack;
  (void)nested;
}

void lambda_order_recovery() {
  auto wrong_order = []<typename T>(T value)
      pre(value > T{}) requires LambdaConstraint<T> {
    // expected-error@-1 {{trailing requires clause must appear before contract specifiers}}
    return value;
  };
  (void)wrong_order;
}
