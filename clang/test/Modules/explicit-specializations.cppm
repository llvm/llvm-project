// Testing that the compiler can select the correct template specialization
// from different template aliasing.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fprebuilt-module-path=%t \
// RUN:     -fsyntax-only -verify

//--- a.cppm

// For template type parameters
export module a;
export template <class C>
struct S {
    static constexpr bool selected = false;
};

export struct A {};

export template <>
struct S<A> {
    static constexpr bool selected = true;
};

export using B = A;

// For template template parameters

export template <template<typename> typename C>
struct V {
    static constexpr bool selected = false;
};

export template <>
struct V<S> {
    static constexpr bool selected = true;
};

// For template non type parameters
export template <int X>
struct Numbers {
    static constexpr bool selected = false;
    static constexpr int value = X;
};

export template<>
struct Numbers<43> {
    static constexpr bool selected = true;
    static constexpr int value = 43;
};

export template <const int *>
struct Pointers {
    static constexpr bool selected = false;
};

export int IntegralValue = 0;
export template<>
struct Pointers<&IntegralValue> {
    static constexpr bool selected = true;
};

export template <void *>
struct NullPointers {
    static constexpr bool selected = false;
};

export template<>
struct NullPointers<nullptr> {
    static constexpr bool selected = true;
};

export template<int (&)[5]>
struct Array {
    static constexpr bool selected = false;
};

export int array[5];
export template<>
struct Array<array> {
    static constexpr bool selected = true;
};

//--- b.cpp
// expected-no-diagnostics
import a;

// Testing for different qualifiers
static_assert(S<B>::selected);
static_assert(S<::B>::selected);
static_assert(::S<B>::selected);
static_assert(::S<::B>::selected);
typedef A C;
static_assert(S<C>::selected);
static_assert(S<::C>::selected);
static_assert(::S<C>::selected);
static_assert(::S<::C>::selected);

namespace D {
    C getAType();
    typedef C E;
}

static_assert(S<D::E>::selected);
static_assert(S<decltype(D::getAType())>::selected);

// Testing we can select the correct specialization for different
// template template argument alising.

static_assert(V<S>::selected);
static_assert(V<::S>::selected);
static_assert(::V<S>::selected);
static_assert(::V<::S>::selected);

// Testing for template non type parameters
static_assert(Numbers<43>::selected);
static_assert(Numbers<21 * 2 + 1>::selected);
static_assert(Numbers<42 + 1>::selected);
static_assert(Numbers<44 - 1>::selected);
static_assert(Numbers<Numbers<43>::value>::selected);
static_assert(!Numbers<44>::selected);

static_assert(Pointers<&IntegralValue>::selected);
static_assert(!Pointers<nullptr>::selected);
static_assert(NullPointers<nullptr>::selected);
static_assert(!NullPointers<(void*)&IntegralValue>::selected);

static_assert(Array<array>::selected);
int another_array[5];
static_assert(!Array<another_array>::selected);
