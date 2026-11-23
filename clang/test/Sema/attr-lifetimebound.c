// RUN: %clang_cc1 -std=c2x -verify %s

int *ptr_param(int *param [[clang::lifetimebound]]);
int *ptr_param_gnu(int *param __attribute__((lifetimebound)));
int *ptr_param_redecl(int *param);
int *ptr_param_redecl(int *param [[clang::lifetimebound]]);
int *ptr_param_redecl(int *param) { return param; }
int *ptr_param_redecl_gnu(int *param);
int *ptr_param_redecl_gnu(int *param __attribute__((lifetimebound)));
int *ptr_param_redecl_gnu(int *param) { return param; }

void void_return(int *param [[clang::lifetimebound]]); // expected-error {{'lifetimebound' attribute cannot be applied to a parameter of a function that returns void; did you mean 'lifetime_capture_by(X)'}}
void void_return_gnu(int *param __attribute__((lifetimebound))); // expected-error {{'lifetimebound' attribute cannot be applied to a parameter of a function that returns void; did you mean 'lifetime_capture_by(X)'}}

int *attr_with_arg(int *param [[clang::lifetimebound(1)]]); // expected-error {{'clang::lifetimebound' attribute takes no arguments}}
int *attr_with_arg_gnu(int *param __attribute__((lifetimebound(1)))); // expected-error {{'lifetimebound' attribute takes no arguments}}

int attr_on_var [[clang::lifetimebound]]; // expected-error {{'clang::lifetimebound' attribute only applies to parameters and implicit object parameters}}
int attr_on_var_gnu __attribute__((lifetimebound)); // expected-error {{'lifetimebound' attribute only applies to parameters and implicit object parameters}}
int * [[clang::lifetimebound]] attr_on_pointee; // expected-error {{'clang::lifetimebound' attribute only applies to parameters and implicit object parameters}}
int (*func_ptr)(int) [[clang::lifetimebound]]; // expected-error {{'clang::lifetimebound' attribute only applies to parameters and implicit object parameters}}
