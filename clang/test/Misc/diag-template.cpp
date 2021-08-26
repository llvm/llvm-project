// RUN: %clang_cc1 -verify %s

namespace default_args {
  template<typename T> struct char_traits;
  template<typename T> struct allocator;
  template<typename T, typename = char_traits<T>, typename = allocator<T>> struct basic_string {};

  typedef basic_string<char> string;

  template<typename T> T f(T);

  void test1() {
    string s;
    f(s).size(); // expected-error {{no member named 'size' in 'default_args::basic_string<char>'}}
  }

  template<typename T> struct default_delete {};
  template<class T, class Deleter = default_delete<T>> class unique_ptr {
  public:
    void f() { T::error(); } // expected-error {{no member named 'error' in 'default_args::basic_string<char>'}}
  };
  template<class T, class Deleter> class unique_ptr<T[], Deleter> {};
  void test2() {
    unique_ptr<string> ups;
    f(ups).reset(); // expected-error {{no member named 'reset' in 'default_args::unique_ptr<default_args::basic_string<char>>'}}
    f(ups).f(); // expected-note {{in instantiation of member function 'default_args::unique_ptr<default_args::basic_string<char>>::f' requested here}}
  }

  template<int A, int B = A> struct Z { int error[B]; }; // expected-error {{negative size}}
  Z<-1> z; // expected-note {{in instantiation of template class 'default_args::Z<-1>' requested here}}

  template<template<typename> class A = allocator, template<typename> class B = A> struct Q {};
  void test3() {
    f(Q<>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<>'}}
    f(Q<allocator>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<>'}}
    f(Q<allocator, allocator>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<>'}}
    f(Q<char_traits>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<default_args::char_traits>'}}
    f(Q<char_traits, char_traits>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<default_args::char_traits>'}}
    f(Q<char_traits, allocator>()).g(); // expected-error {{no member named 'g' in 'default_args::Q<default_args::char_traits, default_args::allocator>'}}
  }
}
