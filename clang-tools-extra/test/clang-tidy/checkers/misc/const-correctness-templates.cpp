// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:   misc-const-correctness.TransformValues: true, \
// RUN:   misc-const-correctness.TransformReferences: true, \
// RUN:   misc-const-correctness.WarnPointersAsValues: false, \
// RUN:   misc-const-correctness.TransformPointersAsValues: false} \
// RUN:   }" -- -fno-delayed-template-parsing

template <typename T>
void type_dependent_variables() {
  T value = 42;
  auto &ref = value;
  T &templateRef = value;

  int value_int = 42;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'value_int' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const value_int
}
void instantiate_template_cases() {
  type_dependent_variables<int>();
  type_dependent_variables<float>();
}

namespace gh57297{
// The expression to check may not be the dependent operand in a dependent
// operator.

// Explicitly adding the conversion operator to int for `Stream` to make the
// explicit instantiation (required due to windows' delayed template parsing
// pre C++20) of `f` work without compile errors. Writing an `operator<<` for
// `Stream` would make the `x << t` expression a CXXOperatorCallExpr, not a
// BinaryOperator.
struct Stream { operator int(); };
template <typename T> void f() { T t; Stream x; x << t; }
void foo() { f<int>(); }
} // namespace gh57297
