// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

struct A;
struct B;

template <typename> constexpr bool True = true;
template <typename T> concept C = True<T>;

void f(C auto &, auto &) = delete;
template <C Q> void f(Q &, C auto &);

void g(struct A *ap, struct B *bp) {
  f(*ap, *bp);
}

template <typename T, typename U> struct X {};

template <typename T, C U, typename V> bool operator==(X<T, U>, V) = delete;
template <C T, C U, C V>               bool operator==(T, X<U, V>);

bool h() {
  return X<void *, int>{} == 0;
}

namespace PR53640 {

template <typename T>
concept C = true;

template <C T>
void f(T t) {} // expected-note {{candidate function [with T = int]}}

template <typename T>
void f(const T &t) {} // expected-note {{candidate function [with T = int]}}

int g() {
  f(0); // expected-error {{call to 'f' is ambiguous}}
}

struct S {
  template <typename T> explicit S(T) noexcept requires C<T> {} // expected-note {{candidate constructor}}
  template <typename T> explicit S(const T &) noexcept {}       // expected-note {{candidate constructor}}
};

int h() {
  S s(4); // expected-error-re {{call to constructor of {{.*}} is ambiguous}}
}

}
