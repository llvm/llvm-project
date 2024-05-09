// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -verify -std=c++14 %s
// RUN: %clang_cc1 -verify -std=c++17 %s
// RUN: %clang_cc1 -verify -std=c++20 %s
// RUN: %clang_cc1 -verify -std=c++23 %s
// RUN: %clang_cc1 -verify -std=c++2c %s

// https://github.com/llvm/llvm-project/issues/10518

template <class T>
class A : public T {
};

template <class T>
class B : public A<T> {
};

template <class T>
class B<int> : public A<T> { // expected-error 0-1 {{}}
	B(T *t) {}
};
