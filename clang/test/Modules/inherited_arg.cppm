// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/A-B.cppm -I%t -emit-module-interface -o %t/A-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/A-C.cppm -I%t -emit-module-interface -o %t/A-C.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

// Test again with reduced BMI.
//
// RUN: %clang_cc1 -std=c++20 %t/A-B.cppm -I%t -emit-reduced-module-interface -o %t/A-B.pcm
// RUN: %clang_cc1 -std=c++20 %t/A-C.cppm -I%t -emit-reduced-module-interface -o %t/A-C.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -fprebuilt-module-path=%t -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only


//--- foo.h
template <typename U, typename T>
class pair {};

template <typename U>
class allocator {};

template <typename T>
class my_traits {};

template <class _Key, class _Tp,
          class _Alloc = allocator<pair<const _Key, _Tp> > >
class unordered_map
{
public:
    unordered_map() {}
};

template<bool, class = void> struct my_enable_if {};
template<class T> struct my_enable_if<true, T> { using type = T; };
template<bool B, class T = void> using my_enable_if_t = typename my_enable_if<B, T>::type;

template<class _InputIterator,
         class _Allocator = allocator<my_traits<_InputIterator>>,
         class = my_enable_if_t<_InputIterator::value>>
unordered_map(_InputIterator, _InputIterator, _Allocator = _Allocator())
  -> unordered_map<my_traits<_InputIterator>, my_traits<_InputIterator>, _Allocator>;

template <class _CharT,
          class _Traits = my_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class basic_string;
typedef basic_string<char, my_traits<char>, allocator<char> > string;

template<class _CharT, class _Traits, class _Allocator>
class basic_string
{
public:
    basic_string();

    template<class _InputIterator, class = my_enable_if_t<_InputIterator::value> > 
            basic_string(_InputIterator __first, _InputIterator __last, const _Allocator& __a);

    void resize(unsigned __n, _CharT __c);
};

extern template void basic_string<char>::resize(unsigned, char);

//--- A-B.cppm
module;
#include "foo.h"
export module A:B;
export using ::string;

//--- A-C.cppm
module;
#include "foo.h"
export module A:C;

//--- A.cppm
export module A;
export import :B;
export import :C;

//--- Use.cpp
import A;
string s;
::unordered_map<int, int> mime_map; // expected-error {{missing '#include'; 'unordered_map' must be declared before it is used}}
                                    // expected-note@* {{declaration here is not visible}}
