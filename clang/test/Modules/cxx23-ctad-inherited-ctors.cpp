// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cpp -fsyntax-only -verify -fprebuilt-module-path=%t

//--- a.cppm
export module a;

template<typename T> struct Base {
  Base(T); 
};

export template<typename T, typename U> struct NotEnoughParams : public Base<T> {
  using Base<T>::Base; 
  NotEnoughParams(T t, U u);
};

// Force deduction guides to be declared here, to ensure they are exported/imported in b.cpp
NotEnoughParams declareDGs(1, 2);

//--- b.cpp
import a;

// Test that we maintain the deduction guide source and source kind
// from Base's deduction guide through a module

NotEnoughParams notEnoughParams(1); // expected-error {{no viable constructor or deduction guide for deduction of template arguments}}

// expected-note@a.cppm:* {{generated from 'Base<T>' constructor}}

// expected-note@a.cppm:* {{candidate template ignored: could not match 'NotEnoughParams<T, U>' against 'int'}} \
// expected-note@a.cppm:* {{implicit deduction guide declared as 'template <typename T, typename U> NotEnoughParams(NotEnoughParams<T, U>) -> NotEnoughParams<T, U>'}} \
// expected-note@a.cppm:* {{candidate function template not viable: requires 2 arguments, but 1 was provided}} \
// expected-note@a.cppm:* {{implicit deduction guide declared as 'template <typename T, typename U> NotEnoughParams(T t, U u) -> NotEnoughParams<T, U>'}}

// expected-note@a.cppm:* {{candidate template ignored: could not deduce template arguments for 'NotEnoughParams<T, U>' from 'Base<T>' [with T = int]}} \
// expected-note@a.cppm:* {{implicit deduction guide declared as 'template <typename T> NotEnoughParams(T) -> typename __ctad_CC_Base_to_NotEnoughParams_0<Base<T>>::type'}} \
// expected-note@a.cppm:* {{candidate template ignored: could not match 'Base<T>' against 'int'}} \
// expected-note@a.cppm:* {{implicit deduction guide declared as 'template <typename T> NotEnoughParams(Base<T>) -> typename __ctad_CC_Base_to_NotEnoughParams_0<Base<T>>::type'}}
