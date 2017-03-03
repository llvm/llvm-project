// RUN: %clang_analyze_cc1 -analyzer-checker=core -std=c++11 -fdelayed-template-parsing -verify %s
// expected-no-diagnostics

template <class T> struct remove_reference      {typedef T type;};
template <class T> struct remove_reference<T&>  {typedef T type;};
template <class T> struct remove_reference<T&&> {typedef T type;};

template <typename T>
typename remove_reference<T>::type&& move(T&& arg) { // this used to crash
  return static_cast<typename remove_reference<T>::type&&>(arg);
}
