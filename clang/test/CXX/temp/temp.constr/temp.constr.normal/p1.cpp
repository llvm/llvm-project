// RUN: %clang_cc1 -std=c++2a -fconcepts-ts -x c++ -verify %s

template<typename T> concept True = true;
template<typename T> concept Foo = True<T*>;
template<typename T> concept Bar = Foo<T&>;
template<typename T> requires Bar<T> struct S { };
template<typename T> requires Bar<T> && true struct S<T> { };

template<typename T> concept True2 = sizeof(T) >= 0;
template<typename T> concept Foo2 = True2<T*>;
// expected-error@-1{{'type name' declared as a pointer to a reference of type 'type-parameter-0-0 &'}}
template<typename T> concept Bar2 = Foo2<T&>;
// expected-note@-1{{while substituting into concept arguments here; substitution failures not allowed in concept arguments}}
template<typename T> requires Bar2<T> struct S2 { };
// expected-note@-1{{template is declared here}}
template<typename T> requires Bar2<T> && true struct S2<T> { };
// expected-error@-1{{class template partial specialization is not more specialized than the primary template}}
// expected-note@-2{{while calculating associated constraint of template 'S2' here}}
