// RUN: %clang_cc1 %s -verify=expected,type-param -std=c++23 -DTYPE_PARAM
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DCONSTANT_PARAM
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DTYPE_TEMPLATE_PARAM
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DDEFAULT_ARG
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DMULTIPLE_PARAMS
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DPARAM_PACK
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DCONSTRAINED_PARAM
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DREQUIRES_CLAUSE
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DNONCLASS_TEMPLATE
// RUN: %clang_cc1 %s -verify=expected,others -std=c++23 -DNONTEMPLATE

namespace std {

#ifdef TYPE_PARAM
template<class> class initializer_list;
// expected-note@-1 2 {{template is declared here}}
#elifdef CONSTANT_PARAM
template<int> class initializer_list;
// expected-error@-1 2 {{std::initializer_list must have a type template parameter}}
#elifdef TYPE_TEMPLATE_PARAM
template<template<class> class> class initializer_list;
// expected-error@-1 2 {{std::initializer_list must have a type template parameter}}
#elifdef DEFAULT_ARG
template<class = int> class initializer_list;
// expected-error@-1 2 {{std::initializer_list cannot have default template arguments}}
#elifdef MULTIPLE_PARAMS
template<class, class> class initializer_list;
// expected-error@-1 2 {{std::initializer_list must have exactly one template parameter}}
#elifdef PARAM_PACK
template<class...> class initializer_list;
// expected-error@-1 2 {{std::initializer_list cannot be a variadic template}}
#elifdef CONSTRAINED_PARAM
template<class> concept C = true;
template<C> class initializer_list;
// expected-error@-1 2 {{std::initializer_list cannot have associated constraints}}
#elifdef REQUIRES_CLAUSE
template<class> requires true class initializer_list;
// expected-error@-1 2 {{std::initializer_list cannot have associated constraints}}
#elifdef NONCLASS_TEMPLATE
template<class> class IL;
template<class T> using initializer_list = IL<T>;
// expected-error@-1 2 {{std::initializer_list must be a class template}}
#elifdef NONTEMPLATE
class initializer_list;
// expected-error@-1 2 {{std::initializer_list must be a class template}}
#else
#error Unexpected test kind
#endif

}

struct Test { // expected-note 2 {{candidate constructor}}
#ifdef CONSTANT_PARAM
    Test(std::initializer_list<1>); // expected-note {{candidate constructor}}
#elifdef TYPE_TEMPLATE_PARAM
    template<class> using A = double;
    Test(std::initializer_list<A>); // expected-note {{candidate constructor}}
#elifdef MULTIPLE_PARAMS
    Test(std::initializer_list<double, double>); // expected-note {{candidate constructor}}
#elifdef NONTEMPLATE
    Test(std::initializer_list); // expected-note {{candidate constructor}}
#else
    Test(std::initializer_list<double>); // expected-note {{candidate constructor}}
#endif
};
Test test {1.2, 3.4}; // expected-error {{no matching constructor}}

auto x = {1};
// type-param-error@-1 {{implicit instantiation of undefined template}}
// others-note@-2 {{used here}}

void f() {
    for(int x : {1, 2});
    // type-param-error@-1 {{implicit instantiation of undefined template}}
    // type-param-error@-2 {{invalid range expression}}
    // others-note@-3 {{used here}}
}
