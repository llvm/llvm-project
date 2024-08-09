// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

template<typename>
concept Constrained = true;

template <typename T>
class C
{
    template<Constrained>
    class D;
};

template <typename T>
template <Constrained>
class C<T>::D
{
};
