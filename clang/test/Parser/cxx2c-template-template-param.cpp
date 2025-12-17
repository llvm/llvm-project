// RUN:  %clang_cc1 -std=c++2c -verify %s

template<template<typename> auto Var>
struct A{};
template<template<auto> auto Var>
struct B{};
template<template<typename> auto Var>
struct C{};
template<template<typename> concept C>
struct D{};
template<template<auto> concept C>
struct E{};

template<template<typename> auto Var>
int V1;
template<template<auto> auto Var>
int V2;
template<template<typename> auto Var>
int V3;
template<template<typename> concept C>
int V4;
template<template<auto> concept C>
int V5;

namespace packs {

template<template<typename> auto... Var>
struct A{};
template<template<auto> auto... Var>
struct B{};
template<template<typename> auto... Var>
struct C{};
template<template<typename> concept... C>
struct D{};
template<template<auto> concept... C>
struct E{};

template<template<typename> auto... Var>
int V1;
template<template<auto> auto...  Var>
int V2;
template<template<typename> auto...  Var>
int V3;
template<template<typename> concept... C>
int V4;
template<template<auto> concept... C>
int V5;

}

namespace concepts {
template<template<auto> concept...>
struct A{};
template<template<auto> concept... C>
struct B{};
template<template<auto> concept& C> // expected-error{{expected identifier}} \
                                    // expected-error {{in declaration of struct 'C'}}
struct C{};
}

namespace vars {
template<template<auto> auto...>
struct A{};
template<template<auto> auto & C> // expected-error {{expected identifier}} \
                                  // expected-error {{extraneous 'template<>'}}
struct B{};
template<template<auto> const auto> // expected-error {{expected identifier}} \
                                    // expected-error {{extraneous 'template<>'}}
struct C{};
}

namespace errors {
template<concept> // expected-error {{expected template parameter}} \
                  // expected-error {{extraneous 'template<>' in declaration of struct 'A'}}
struct A{};
template<template<concept> auto> // expected-error {{expected template parameter}} \
                                 // expected-error {{template template parameter must have its own template parameters}}
struct B{};
}
