// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-arc-cxxlib=libstdc++ -fobjc-runtime-has-weak -verify %s
// expected-no-diagnostics

@interface A @end

int check0[std::__is_scalar<__strong id>::__value? -1 : 1];
int check1[std::__is_scalar<__weak id>::__value? -1 : 1];
int check2[std::__is_scalar<__autoreleasing id>::__value? -1 : 1];
int check3[std::__is_scalar<__strong A*>::__value? -1 : 1];
int check4[std::__is_scalar<__weak A*>::__value? -1 : 1];
int check5[std::__is_scalar<__autoreleasing A*>::__value? -1 : 1];
