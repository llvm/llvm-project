// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

namespace mystd {
inline namespace bar {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T>
[[clang::behaves_like_std("moved")]] typename remove_reference<T>::type &&move(T &&t); // expected-error {{not a valid std builtin for attribute 'behaves_like_std'}}

template <class T>
[[clang::behaves_like_std("__builtin_abs")]] typename remove_reference<T>::type &&move2(T &&t); // expected-error {{not a valid std builtin for attribute 'behaves_like_std'}}

template <class T>
[[clang::behaves_like_std("strlen")]] typename remove_reference<T>::type &&move3(T &&t); // expected-error {{not a valid std builtin for attribute 'behaves_like_std'}}

}
}

