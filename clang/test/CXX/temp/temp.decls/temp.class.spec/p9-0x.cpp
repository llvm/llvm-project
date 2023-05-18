// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// -- The argument list of the specialization shall not be identical
//    to the implicit argument list of the primary template.

template<typename T, typename ...Types>
struct X1; 

template<typename T, typename ...Types>
struct X1<T, Types...> // expected-error{{class template partial specialization does not specialize any template argument; to define the primary template, remove the template argument list}}
{ };


