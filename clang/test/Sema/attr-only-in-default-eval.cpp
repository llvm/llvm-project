// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef float float_t [[clang::available_only_in_default_eval_method]];
using double_t __attribute__((available_only_in_default_eval_method)) = double;

// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
class  __attribute__((available_only_in_default_eval_method)) C1 {
};
// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
class  [[clang::available_only_in_default_eval_method]] C2 {
};

// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
struct [[clang::available_only_in_default_eval_method]] S1;
// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
struct __attribute__((available_only_in_default_eval_method)) S2;

// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
void __attribute__((available_only_in_default_eval_method)) foo();
// expected-error@+1{{'available_only_in_default_eval_method' attribute cannot be applied to types}}
void [[clang::available_only_in_default_eval_method]] goo();
// expected-error@+1{{'available_only_in_default_eval_method' attribute cannot be applied to types}}
void bar() [[clang::available_only_in_default_eval_method]];
// expected-error@+1{{'available_only_in_default_eval_method' attribute only applies to typedefs}}
void barz() __attribute__((available_only_in_default_eval_method));

