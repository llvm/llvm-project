// RUN: %clang_cc1 -std=c2x -verify %s

struct S {
  int *x;
};

void capture(int *x [[clang::lifetime_capture_by(s, t)]],
             struct S *s,
             struct S *t);
void capture_gnu(int *x __attribute__((lifetime_capture_by(s, t))),
                 struct S *s,
                 struct S *t);

[[clang::lifetime_capture_by(s)]] // expected-error {{'clang::lifetime_capture_by' attribute only applies to parameters and implicit object parameters}}
void attr_on_function(int *x);
void attr_on_function_gnu(int *x) __attribute__((lifetime_capture_by(s))); // expected-error {{'lifetime_capture_by' attribute only applies to parameters and implicit object parameters}}

void invalid_args(int *x1 [[clang::lifetime_capture_by(12345 + 12)]], // expected-error {{'lifetime_capture_by' attribute argument '12345 + 12' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                  int *x2 [[clang::lifetime_capture_by(no_such_param)]], // expected-error {{'lifetime_capture_by' attribute argument 'no_such_param' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                  int *x3 [[clang::lifetime_capture_by("no_such_param")]], // expected-error {{'lifetime_capture_by' attribute argument '"no_such_param"' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                  int *x4 [[clang::lifetime_capture_by()]], // expected-error {{'lifetime_capture_by' attribute specifies no capturing entity}}
                  int *x5 [[clang::lifetime_capture_by(x5)]]); // expected-error {{'lifetime_capture_by' argument references itself}}

void invalid_args_gnu(int *x1 __attribute__((lifetime_capture_by(12345 + 12))), // expected-error {{'lifetime_capture_by' attribute argument '12345 + 12' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                      int *x2 __attribute__((lifetime_capture_by(no_such_param))), // expected-error {{'lifetime_capture_by' attribute argument 'no_such_param' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                      int *x3 __attribute__((lifetime_capture_by("no_such_param"))), // expected-error {{'lifetime_capture_by' attribute argument '"no_such_param"' is not a known function parameter; must be a function parameter, 'this', 'global' or 'unknown'}}
                      int *x4 __attribute__((lifetime_capture_by())), // expected-error {{'lifetime_capture_by' attribute specifies no capturing entity}}
                      int *x5 __attribute__((lifetime_capture_by(x5)))); // expected-error {{'lifetime_capture_by' argument references itself}}
