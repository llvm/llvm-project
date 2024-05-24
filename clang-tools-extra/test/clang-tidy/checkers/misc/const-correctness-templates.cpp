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

// Explicitly not declaring a (templated) stream operator
// so the `<<` is a `binaryOperator` with a dependent type.
struct Stream { };
template <typename T> void f() { T t; Stream x; x << t; }
} // namespace gh57297

namespace gh70323{
// A fold expression may contain the checked variable as it's initializer.
// We don't know if the operator modifies that variable because the
// operator is type dependent due to the parameter pack.

struct Stream {};
template <typename... Args>
void concatenate1(Args... args)
{
    Stream stream;
    (stream << ... << args);
}

template <typename... Args>
void concatenate2(Args... args)
{
    Stream stream;
    (args << ... << stream);
}

template <typename... Args>
void concatenate3(Args... args)
{
    Stream stream;
    (..., (stream << args));
}
} // namespace gh70323

namespace gh60895 {

template <class T> void f1(T &&a);
template <class T> void f2(T &&a);
template <class T> void f1(T &&a) { f2<T>(a); }
template <class T> void f2(T &&a) { f1<T>(a); }
void f() {
  int x = 0;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'x' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const x = 0;
  f1(x);
}

} // namespace gh60895
