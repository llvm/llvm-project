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
  T &templateRef = value;

  // 'auto &ref' deduces to a dependent type, so the variable is not analyzed
  // inside the template instantiation (the deduced type may differ between
  // instantiations).
  auto &ref = value;

  int value_int = 42;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'value_int' of type 'int' can be declared 'const'
  // CHECK-FIXES: int const value_int = 42;

  // Only dependent 'auto &' references are excluded. 'auto' variables deduced
  // from a non-dependent initializer are still analyzed (here via the
  // uninstantiated pattern), as are 'auto' values and 'auto' pointers.
  int concrete = 42;
  auto &ref_concrete = concrete;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'ref_concrete' of type 'int &' can be declared 'const'
  // CHECK-FIXES: auto  const&ref_concrete = concrete;
  auto val_concrete = concrete;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'val_concrete' of type 'int' can be declared 'const'
  // CHECK-FIXES: auto const val_concrete = concrete;
  auto *ptr_concrete = &concrete;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: pointee of variable 'ptr_concrete' of type 'int *' can be declared 'const'
  // CHECK-FIXES: auto  const*ptr_concrete = &concrete;

  // A typedef to a template parameter keeps the substituted parameter visible
  // in the type sugar, so it is excluded both as a reference and as a value:
  // the constness of such a variable can differ between instantiations.
  using TypedefToTemplate = T;
  TypedefToTemplate &td_ref = value;
  TypedefToTemplate td_val = value;
  (void)td_val;

  // Pointers are still analyzed even when the pointee derives from a template
  // parameter: the suggestion concerns the pointer/pointee spelling, like the
  // bare 'T *' case, not member constness.
  TypedefToTemplate *td_ptr = &value;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: pointee of variable 'td_ptr' of type 'TypedefToTemplate *' (aka 'int *') can be declared 'const'
  // CHECK-FIXES: TypedefToTemplate  const*td_ptr = &value;
  (void)*td_ptr;

  // A typedef to a concrete type is not template-derived, so such references
  // are still analyzed.
  using ConcreteAlias = int;
  ConcreteAlias &concrete_alias_ref = concrete;
  // CHECK-MESSAGES:[[@LINE-1]]:3: warning: variable 'concrete_alias_ref' of type 'ConcreteAlias &' (aka 'int &') can be declared 'const'
  // CHECK-FIXES: ConcreteAlias  const&concrete_alias_ref = concrete;
}
void instantiate_template_cases() {
  type_dependent_variables<int>();
  type_dependent_variables<float>();
}

// Class template member functions are instantiations too: a dependent
// 'auto &' reference must be excluded there as well, while non-dependent
// 'auto' variables stay analyzed.
template <typename T>
struct ClassTemplate {
  int method() {
    T value{};
    auto &dependent_ref = value;

    int c1 = 42;
    // CHECK-MESSAGES:[[@LINE-1]]:5: warning: variable 'c1' of type 'int' can be declared 'const'
    // CHECK-FIXES: int const c1 = 42;
    auto &concrete_ref = c1;
    // CHECK-MESSAGES:[[@LINE-1]]:5: warning: variable 'concrete_ref' of type 'int &' can be declared 'const'
    // CHECK-FIXES: auto  const&concrete_ref = c1;
    int c2 = 42;
    // CHECK-MESSAGES:[[@LINE-1]]:5: warning: variable 'c2' of type 'int' can be declared 'const'
    // CHECK-FIXES: int const c2 = 42;
    auto concrete_val = c2;
    // CHECK-MESSAGES:[[@LINE-1]]:5: warning: variable 'concrete_val' of type 'int' can be declared 'const'
    // CHECK-FIXES: auto const concrete_val = c2;
    int c3 = 42;
    auto *concrete_ptr = &c3;
    // CHECK-MESSAGES:[[@LINE-1]]:5: warning: pointee of variable 'concrete_ptr' of type 'int *' can be declared 'const'
    // CHECK-FIXES: auto  const*concrete_ptr = &c3;
    return concrete_ref + concrete_val + *concrete_ptr +
           static_cast<int>(sizeof(dependent_ref));
  }
};
template struct ClassTemplate<int>;

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
