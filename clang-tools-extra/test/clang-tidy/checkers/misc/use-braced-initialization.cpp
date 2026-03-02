// RUN: %check_clang_tidy -std=c++11-or-later %s misc-use-braced-initialization %t \
// RUN:   -- -- -I %S/../Inputs/Headers

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

void basic_multiple_args() {
  Simple w(1, 2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w{1, 2.0};
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
    break;
  }
}

void expression_arg() {
  Simple w(1 + 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w{1 + 2};
}

void variable_arg(int x) {
  Simple w(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w{x};
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
  (void)p;
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
  (void)p;
}

void class_comment_before_parens() {
  Simple w /*comment*/ (1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use braced initialization
  // CHECK-FIXES: Simple w /*comment*/ {1};
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
  (void)p;
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

void force_instantiation() {
  template_instantiated<Simple>(1);
  template_instantiated2<Simple>(1);
  template_temporary_single<Simple>();
  template_temporary_multi<Simple>();
  (void)template_return<Simple>();
  delete template_new_expr<Simple>();
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

struct InitListMutableRef {
  InitListMutableRef(std::initializer_list<int> &);
  InitListMutableRef(int);
};

void il_mutable_ref() {
  InitListMutableRef x(1);
}

struct InitListRvalueRef {
  InitListRvalueRef(std::initializer_list<int> &&);
  InitListRvalueRef(int);
};

void il_rvalue_ref() {
  InitListRvalueRef x(1);
}

struct InitListVolatileRef {
  InitListVolatileRef(volatile std::initializer_list<int> &);
  InitListVolatileRef(int);
};

void il_volatile_ref() {
  InitListVolatileRef x(1);
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
