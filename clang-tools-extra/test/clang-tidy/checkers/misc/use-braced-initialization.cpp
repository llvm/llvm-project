// RUN: %check_clang_tidy -std=c++11-or-later %s misc-use-braced-initialization %t -- --fix-notes

#include <string>
#include <vector>

struct Simple {
  Simple(int);
  Simple(int, double);
  Simple(const Simple &);
};

struct Explicit {
  explicit Explicit(int);
};

struct Aggregate {
  int a, b;
};

struct Takes {
  Takes(Aggregate);
};

struct TwoAggregates {
  TwoAggregates(Aggregate, Aggregate);
};

struct WithDefault {
  WithDefault(int, int = 0);
};

struct HasDefault {
  HasDefault();
};

struct Outer {
  struct Inner {
    Inner(int);
  };
};

namespace ns {
struct Ns {
  Ns(int);
};
} // namespace ns

#define MAKE_SIMPLE(x) Simple w(x)
#define WRAP_PARENS(x) (x)
#define TYPE_ALIAS Simple

void basic_single_arg() {
  Simple w(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization instead of parenthesized initialization [misc-use-braced-initialization]
  // CHECK-FIXES: Simple w{1};
}

void explicit_ctor() {
  Explicit e(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: Explicit e{42};
}

void copy_construction() {
  Simple w1(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w1{1};
  Simple w2(w1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w2{w1};
}

void static_local() {
  static Simple sw(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization
  // CHECK-FIXES: static Simple sw{1};
}

void const_variable() {
  const Simple cw(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use braced initialization
  // CHECK-FIXES: const Simple cw{1};
}

void default_args_ctor() {
  WithDefault m(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
  // CHECK-FIXES: WithDefault m{1};
}

void nested_type() {
  Outer::Inner oi(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use braced initialization
  // CHECK-FIXES: Outer::Inner oi{1};
}

void namespaced_type() {
  ns::Ns g(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: ns::Ns g{1};
}

void for_loop_init() {
  for (Simple fw(1);;) {
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
    // CHECK-FIXES: for (Simple fw{1};;) {
  }
}

void if_init_statement() {
  if (Simple s(1); true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use braced initialization
    // CHECK-FIXES: if (Simple s{1}; true) {
  }
}

void switch_init_statement(int x) {
  switch (Simple s(x); x) {
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use braced initialization
    // CHECK-FIXES: switch (Simple s{x}; x) {
  }
}

void ternary_arg(bool c) {
  Simple s(c ? 1 : 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple s{c ? 1 : 2};
}

void multi_decl_class() {
  Simple a(1), b(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use braced initialization
  // CHECK-FIXES: Simple a{1}, b{2};
}

Simple global_simple(1);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
// CHECK-FIXES: Simple global_simple{1};

namespace ns_scope {
Simple ns_var(2);
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
// CHECK-FIXES: Simple ns_var{2};
} // namespace ns_scope

// Macro wraps only the type name; parens are in user code, safe to fix.
void macro_type_only() {
  TYPE_ALIAS w(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use braced initialization
  // CHECK-FIXES: TYPE_ALIAS w{1};
}

void scalar_int() {
  int x(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int x{42};
}

void scalar_double() {
  double d(3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: double d{3.14};
}

void scalar_expression(int a) {
  int y(a + 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int y{a + 1};
}

void scalar_pointer() {
  int *p(nullptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: int *p{nullptr};
}

void multi_decl_scalar() {
  int a(1), b(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: use braced initialization
  // CHECK-FIXES: int a{1}, b{2};
}

void temporary_single_arg() {
  Simple(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use braced initialization
  // CHECK-FIXES: Simple{1};
}

void temporary_multi_arg() {
  Simple(1, 2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use braced initialization
  // CHECK-FIXES: Simple{1, 2.0};
}

void copy_init_rhs() {
  Simple w = Simple(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use braced initialization
  // CHECK-FIXES: Simple w = Simple{1};
}

void auto_copy_init() {
  auto w = Simple(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: auto w = Simple{1};
}

Simple return_simple() {
  return Simple(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: return Simple{1};
}

void func_arg(Simple);
void simple_as_argument() {
  func_arg(Simple(1));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: func_arg(Simple{1});
}

void new_multi_arg() {
  Simple *p = new Simple(1, 2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use braced initialization
  // CHECK-FIXES: Simple *p = new Simple{1, 2.0};
}

void braced_arg() {
  Takes tp({1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: Takes tp{{[{][{]}}1, 2{{[}][}]}};
}

void braced_constructed_arg() {
  Takes tp(Aggregate{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: Takes tp{Aggregate{1, 2}};
}

void multiple_braced_args() {
  TwoAggregates t({1, 2}, {3, 4});
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization
  // CHECK-FIXES: TwoAggregates t{{[{][{]}}1, 2}, {3, 4{{[}][}]}};
}

void temporary_braced_arg() {
  (void)Takes({1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)Takes{{[{][{]}}1, 2{{[}][}]}};
}

void new_braced_arg() {
  Takes *p = new Takes({1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use braced initialization
  // CHECK-FIXES: Takes *p = new Takes{{[{][{]}}1, 2{{[}][}]}};
}

void class_comment_inside_parens() {
  Simple w(/*comment*/ 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w{/*comment*/ 1};
}

void scalar_comment_before_parens() {
  int x /*comment*/ (42);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int x /*comment*/ {42};
}

void scalar_comment_inside_parens() {
  int x(/*comment*/ 42);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int x{/*comment*/ 42};
}

void scalar_comment_after_init() {
  int x(42 /*comment*/);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-FIXES: int x{42 /*comment*/};
}

struct L1 {
  int a, b;
  L1(int, int);
};

struct L2 {
  L1 x;
  int y;
  L2(L1, int);
};

struct L3 {
  L2 m;
  int z;
  L3(L2, int);
};

struct L4 {
  L3 n;
  int w;
  L4(L3, int);
};

void nested_ctors_two_levels() {
  L2 v(L1(1, 2), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-FIXES: L2 v{L1{1, 2}, 3};
}

void nested_ctors_three_levels() {
  L3 v(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void nested_ctors_four_levels() {
  L4 v(L3(L2(L1(1, 2), 3), 4), 5);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:11: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-4]]:14: warning: use braced initialization
  // CHECK-FIXES: L4 v{L3{L2{L1{1, 2}, 3}, 4}, 5};
}

void nested_ctors_temporary() {
  (void)L3(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:15: warning: use braced initialization
  // CHECK-FIXES: (void)L3{L2{L1{1, 2}, 3}, 4};
}

void nested_ctors_new() {
  L3 *p = new L3(L2(L1(1, 2), 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-3]]:21: warning: use braced initialization
  // CHECK-FIXES: L3 *p = new L3{L2{L1{1, 2}, 3}, 4};
}

// Mixed: some levels already braced, only paren levels get fixed.
void nested_ctors_mixed() {
  L3 v(L2{L1(1, 2), 3}, 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void nested_ctors_mixed_inner_braced() {
  L3 v(L2(L1{1, 2}, 3), 4);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:8: warning: use braced initialization
  // CHECK-FIXES: L3 v{L2{L1{1, 2}, 3}, 4};
}

void already_braced() {
  Simple w{1};
}

void already_braced_temporary() {
  Simple{1};
}

void new_already_braced() {
  Simple *p = new Simple{1};
  (void)p;
}

void scalar_already_braced() {
  int x{42};
}

void direct_auto() {
  auto w(1);
}

void scalar_auto() {
  auto x(42);
}

void scalar_copy_init() {
  int x = 42;
}

void default_construction() {
  HasDefault d;
}

void macro_full_decl() {
  MAKE_SIMPLE(1);
}

void macro_wraps_parens() {
  Simple w WRAP_PARENS(1);
}

template <typename T>
void template_dependent() {
  T t(1);
}

template <typename T>
void template_instantiated(int x) {
  T t(x);
}

template <typename T>
void template_instantiated2(T x) {
  auto t(x);
}

template <typename T>
void template_temporary_single() {
  (void)T(1);
}

template <typename T>
void template_temporary_multi() {
  (void)T(1, 2.0);
}

template <typename T>
T template_return() {
  return T(1);
}

template <typename T>
T *template_new_expr() {
  return new T(1);
}

template <typename T, typename... Args>
void template_variadic(Args... args) {
  T t(args...);
}

void force_instantiation() {
  template_instantiated<Simple>(1);
  template_instantiated2<Simple>(1);
  template_temporary_single<Simple>();
  template_temporary_multi<Simple>();
  (void)template_return<Simple>();
  delete template_new_expr<Simple>();
  template_variadic<Simple>(1);
}

struct InitListByValue {
  InitListByValue(std::initializer_list<int>);
  InitListByValue(int);
  InitListByValue(int, int);
};

void il_by_value() {
  InitListByValue x(1);
}

void il_by_value_multi() {
  InitListByValue x(1, 2);
}

struct InitListConstRef {
  InitListConstRef(const std::initializer_list<int> &);
  InitListConstRef(int);
};

void il_const_ref() {
  InitListConstRef x(1);
}

struct InitListConstVolatileRef {
  InitListConstVolatileRef(const volatile std::initializer_list<int> &);
  InitListConstVolatileRef(int);
};

void il_const_volatile_ref() {
  InitListConstVolatileRef x(1);
}

struct InitListDefaults {
  InitListDefaults(std::initializer_list<int>, int = 0, double = 1.0);
  InitListDefaults(int);
};

void il_other_params_defaulted() {
  InitListDefaults x(1);
}

void il_std_string() {
  std::string s("hello");
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
  // CHECK-FIXES: std::string s{"hello"};
}

void il_std_string_count_char() {
  std::string s(3, 'a');
}

void il_std_vector() {
  std::vector<int> v(5);
}

void il_std_vector_count_value() {
  std::vector<int> v(5, 1);
}

void il_std_vector_temporary() {
  std::vector<int>(5);
}

void il_std_vector_already_braced() {
  std::vector<int> v{1, 2, 3};
}

void il_new_std_vector() {
  std::vector<int> *p = new std::vector<int>(5);
  (void)p;
}

void il_braced_arg() {
  InitListByValue x({1, 2, 3});
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use braced initialization
  // CHECK-FIXES: InitListByValue x{{[{][{]}}1, 2, 3{{[}][}]}};
}

struct InitListSecondParam {
  InitListSecondParam(int, std::initializer_list<int>);
  InitListSecondParam(int, int);
};

void il_not_first_param() {
  InitListSecondParam x(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use braced initialization
  // CHECK-FIXES: InitListSecondParam x{1, 2};
}

struct InitListPointer {
  InitListPointer(std::initializer_list<int> *);
  InitListPointer(int);
};

void il_pointer() {
  InitListPointer x(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use braced initialization
  // CHECK-FIXES: InitListPointer x{1};
}

struct InitListNoDefaults {
  InitListNoDefaults(std::initializer_list<int>, int);
  InitListNoDefaults(int);
};

void il_other_params_no_defaults() {
  InitListNoDefaults x(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use braced initialization
  // CHECK-FIXES: InitListNoDefaults x{1};
}

struct TemplateCtor {
  TemplateCtor(int);
  template <class T> TemplateCtor(T);
};

void il_template_ctor() {
  TemplateCtor x(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use braced initialization
  // CHECK-FIXES: TemplateCtor x{1};
}

namespace custom {
template <typename T> class initializer_list {};
} // namespace custom

struct HasCustomInitList {
  HasCustomInitList(custom::initializer_list<int>);
  HasCustomInitList(int);
};

void il_custom_namespace_not_std() {
  // custom::initializer_list is not in std, so it doesn't block conversion.
  HasCustomInitList x(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use braced initialization
  // CHECK-FIXES: HasCustomInitList x{42};
}

struct RecordArg {};

struct InitListRecordToNonRecord {
  InitListRecordToNonRecord(std::initializer_list<int>);
  InitListRecordToNonRecord(RecordArg);
};

void il_record_arg_non_record_element() {
  RecordArg r;
  InitListRecordToNonRecord x(r);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use braced initialization
  // CHECK-FIXES: InitListRecordToNonRecord x{r};
}

struct InitListPointerArgArithmetic {
  InitListPointerArgArithmetic(std::initializer_list<int>);
  InitListPointerArgArithmetic(int *);
};

void il_pointer_arg_arithmetic_element() {
  int v = 0;
  InitListPointerArgArithmetic x(&v);
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use braced initialization
  // CHECK-FIXES: InitListPointerArgArithmetic x{&v};
}

struct InitListPointerArgBool {
  InitListPointerArgBool(std::initializer_list<bool>);
  InitListPointerArgBool(int *);
};

void il_pointer_arg_bool_element() {
  int v = 0;
  InitListPointerArgBool x(&v);
}

struct InitListMixedArgs {
  InitListMixedArgs(std::initializer_list<int>);
  InitListMixedArgs(int, RecordArg);
};

void il_mixed_args_second_no_convert() {
  RecordArg r;
  InitListMixedArgs x(1, r);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use braced initialization
  // CHECK-FIXES: InitListMixedArgs x{1, r};
}

enum class ScopedKey { A };
struct InitListVsScopedEnum {
  InitListVsScopedEnum(std::initializer_list<int>);
  InitListVsScopedEnum(ScopedKey);
};
void il_scoped_enum_arg() {
  InitListVsScopedEnum x(ScopedKey::A);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use braced initialization
  // CHECK-FIXES: InitListVsScopedEnum x{ScopedKey::A};
}

struct NoConvOps {};
struct InitListVsNoConvOps {
  InitListVsNoConvOps(std::initializer_list<int>);
  InitListVsNoConvOps(NoConvOps);
};
void il_noconv_class_arg() {
  NoConvOps r;
  InitListVsNoConvOps x(r);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use braced initialization
  // CHECK-FIXES: InitListVsNoConvOps x{r};
}

struct ConvertsToInt {
  operator int() const;
};
struct InitListVsConvClass {
  InitListVsConvClass(std::initializer_list<int>);
  InitListVsConvClass(ConvertsToInt);
};
void il_conv_class_arg() {
  ConvertsToInt c;
  InitListVsConvClass x(c);
}

struct OwnerOfSimple {
  Simple s;
  OwnerOfSimple() : s(1)
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use braced initialization
  // CHECK-FIXES: OwnerOfSimple() : s{1}
  {}
};

struct OwnerOfScalar {
  int n;
  OwnerOfScalar() : n(42)
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use braced initialization
  // CHECK-FIXES: OwnerOfScalar() : n{42}
  {}
};

struct OwnerAlreadyBraced {
  Simple s;
  OwnerAlreadyBraced() : s{1} {}
};

void scalar_functional_cast() {
  (void)int(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)int{42};
}

int scalar_functional_cast_return(int x) {
  return int(x + 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: return int{x + 1};
}

void new_scalar() {
  int *p = new int(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-FIXES: int *p = new int{42};
}

void lambda_body_init() {
  auto f = [](int x) {
    Simple s(x);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
    // CHECK-FIXES: Simple s{x};
  };
}

// Type aliases.
typedef Simple SimpleTypedef;
using SimpleUsing = Simple;

void type_alias_typedef() {
  SimpleTypedef s(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization
  // CHECK-FIXES: SimpleTypedef s{1};
}

void type_alias_using() {
  SimpleUsing s(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use braced initialization
  // CHECK-FIXES: SimpleUsing s{1};
}

// Constexpr variables.
void constexpr_init() {
  constexpr int x(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization
  // CHECK-FIXES: constexpr int x{42};
}

// Narrowing conversions: warning emitted but no fix-it, with note.
struct NarrowingTarget {
  NarrowingTarget(short);
};

void narrowing_float_to_int() {
  int x(3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:9: note: narrowing conversion from 'double' to 'int'
  // CHECK-FIXES: int x{3.14};
}

void narrowing_int_to_short(int n) {
  short s(n);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: note: narrowing conversion from 'int' to 'short'
  // CHECK-FIXES: short s{n};
}

void narrowing_int_to_short_constant() {
  short s(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: short s{1};
}

void narrowing_double_to_float(double d) {
  float f(d);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: note: narrowing conversion from 'double' to 'float'
  // CHECK-FIXES: float f{d};
}

void narrowing_double_to_float_constant() {
  float f(1.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: float f{1.0};
}

void narrowing_double_to_float_constant_overflow() {
  float f(1e300);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: note: narrowing conversion from 'double' to 'float'
  // CHECK-FIXES: float f{1e300};
}

void narrowing_int_to_float(int n) {
  float f(n);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: note: narrowing conversion from 'int' to 'float'
  // CHECK-FIXES: float f{n};
}

void narrowing_int_to_float_constant() {
  float f(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: float f{1};
}

void narrowing_int_to_float_constant_inexact() {
  float f(16777217); // not exactly representable in float
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:11: note: narrowing conversion from 'int' to 'float'
  // CHECK-FIXES: float f{16777217}; // not exactly representable in float
}

void narrowing_ctor_arg(int n) {
  NarrowingTarget t(n);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:21: note: narrowing conversion from 'int' to 'short'
  // CHECK-FIXES: NarrowingTarget t{n};
}

void narrowing_ctor_arg_constant() {
  NarrowingTarget t(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use braced initialization
  // CHECK-FIXES: NarrowingTarget t{1};
}

void narrowing_functional_cast() {
  (void)int(3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:13: note: narrowing conversion from 'double' to 'int'
  // CHECK-FIXES: (void)int{3.14};
}

void narrowing_new_scalar() {
  int *p = new int(3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:20: note: narrowing conversion from 'double' to 'int'
  // CHECK-FIXES: int *p = new int{3.14};
}

struct NarrowingMember {
  short n;
  NarrowingMember(int x) : n(x) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:30: note: narrowing conversion from 'int' to 'short'
  // CHECK-FIXES: NarrowingMember(int x) : n{x} {}
};

void narrowing_ptr_to_bool() {
  int *p = nullptr;
  bool b(p);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:10: note: narrowing conversion
  // CHECK-FIXES: bool b{p};
}

struct BoolCtor {
  BoolCtor(bool);
};
void narrowing_ptr_to_bool_ctor() {
  int *p = nullptr;
  BoolCtor bc(p);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:15: note: narrowing conversion
  // CHECK-FIXES: BoolCtor bc{p};
}

void narrowing_signed_neg_to_unsigned() {
  unsigned u(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:14: note: narrowing conversion from 'int' to 'unsigned int'
  // CHECK-FIXES: unsigned u{-1};
}

void no_narrowing_int_to_bool() {
  bool b(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: bool b{0};
}

struct TwoNarrowing {
  TwoNarrowing(short, short);
};
void narrowing_multiple_args(int a, int b) {
  TwoNarrowing t(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:18: note: narrowing conversion from 'int' to 'short'
  // CHECK-MESSAGES: :[[@LINE-3]]:21: note: narrowing conversion from 'int' to 'short'
  // CHECK-FIXES: TwoNarrowing t{a, b};
}

void reference_init() {
  int a = 1;
  int &r(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: int &r{a};
}

void volatile_variable() {
  volatile int x(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use braced initialization
  // CHECK-FIXES: volatile int x{42};
}

struct BaseClass {
  BaseClass(int);
};

struct DerivedFromBase : BaseClass {
  DerivedFromBase() : BaseClass(1)
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use braced initialization
  // CHECK-FIXES: DerivedFromBase() : BaseClass{1}
  {}
};

struct DelegatingCtor {
  DelegatingCtor(int);
  DelegatingCtor() : DelegatingCtor(1)
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use braced initialization
  // CHECK-FIXES: DelegatingCtor() : DelegatingCtor{1}
  {}
};

struct DelegatingNarrowing {
  DelegatingNarrowing(short);
  DelegatingNarrowing(int x, int) : DelegatingNarrowing(x)
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use braced initialization
  // CHECK-MESSAGES: :[[@LINE-2]]:57: note: narrowing conversion from 'int' to 'short'
  // CHECK-FIXES: DelegatingNarrowing(int x, int) : DelegatingNarrowing{x}
  {}
};

enum Color { Red, Green, Blue };
void enum_functional_cast() {
  (void)Color(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)Color{0};
}

enum class ScopedColor { R, G, B };
void scoped_enum_cast() {
  (void)ScopedColor(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)ScopedColor{0};
}

struct WithStatic {
  static Simple member;
};
Simple WithStatic::member(1);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use braced initialization
// CHECK-FIXES: Simple WithStatic::member{1};

void zero_arg_temporary() {
  (void)Simple(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: (void)Simple{1};
}

struct BaseWithIL {
  BaseWithIL(std::initializer_list<int>);
  BaseWithIL(int, int);
};
struct DerivedNoUsing : BaseWithIL {
  DerivedNoUsing(int a, int b) : BaseWithIL(a, b) {}
};

struct BaseNoIL {
  BaseNoIL(int, int);
};
struct DerivedFromNoIL : BaseNoIL {
  DerivedFromNoIL(int a, int b) : BaseNoIL(a, b) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use braced initialization
  // CHECK-FIXES: DerivedFromNoIL(int a, int b) : BaseNoIL{a, b} {}
};

struct BitfieldStruct {
  int x : 3;
  BitfieldStruct(int n) : x(n)
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use braced initialization
  // CHECK-FIXES: BitfieldStruct(int n) : x{n}
  {}
};

