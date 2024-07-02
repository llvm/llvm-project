// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-arc-cxxlib=libstdc++ -fobjc-runtime-has-weak -verify %s

@interface A @end

int check0[std::__is_scalar<__strong id>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
int check1[std::__is_scalar<__weak id>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
int check2[std::__is_scalar<__autoreleasing id>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
int check3[std::__is_scalar<__strong A*>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
int check4[std::__is_scalar<__weak A*>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
int check5[std::__is_scalar<__autoreleasing A*>::__value? -1 : 1];
// expected-warning@-1 {{using the name of the builtin '__is_scalar' outside of a builtin invocation is deprecated}}
