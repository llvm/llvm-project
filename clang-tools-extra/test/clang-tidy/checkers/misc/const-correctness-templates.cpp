// RUN: %check_clang_tidy %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: [\
// RUN:   {key: 'misc-const-correctness.TransformValues', value: true}, \
// RUN:   {key: 'misc-const-correctness.TransformReferences', value: true}, \
// RUN:   {key: 'misc-const-correctness.WarnPointersAsValues', value: false}, \
// RUN:   {key: 'misc-const-correctness.TransformPointersAsValues', value: false}, \
// RUN:   ]}" -- -fno-delayed-template-parsing

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
