// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 M.cpp -emit-module-interface -o M.pcm
// RUN: %clang_cc1 -std=c++20 N.cpp -emit-module-interface -o N.pcm \
// RUN:   -fmodule-file=M=M.pcm
// RUN: %clang_cc1 -std=c++20 Q.cpp -emit-module-interface -o Q.pcm
// RUN: %clang_cc1 -std=c++20 Q-impl.cpp -fsyntax-only -fmodule-file=Q=Q.pcm \
// RUN:   -fmodule-file=N=N.pcm -fmodule-file=M=M.pcm -verify

//--- M.cpp
export module M;
namespace R {
export struct X {};
export void f(X);
} // namespace R
namespace S {
export void f(R::X, R::X);
}

//--- N.cpp
export module N;
import M;
export R::X make();
namespace R {
static int g(X);
}
export template <typename T, typename U>
void apply(T t, U u) {
  f(t, u);
  g(t);
}

//--- Q.cpp
export module Q;

//--- Q-impl.cpp
module Q;
import N;

namespace S {
struct Z {
  template <typename T> operator T();
};
} // namespace S
void test() {
  // OK, decltype(x) is R::X in module M
  auto x = make();

  // error: R and R::f are not visible here
  R::f(x); // expected-error {{no type named 'f' in namespace 'R'}}

  f(x); // Found by [basic.lookup.argdep] / p4.3

  // error: S::f in module M not considered even though S is an associated
  // namespace, since the entity Z is in a different module from f.
  f(x, S::Z()); // expected-error {{no matching function for call to 'f'}}
  // expected-note@M.cpp:4 {{candidate function not viable: requires 1 argument, but 2 were provided}}

  // error: S::f is visible in instantiation context, but  R::g has internal
  // linkage and cannot be used outside N.cpp
  apply(x, S::Z()); // expected-error@N.cpp:10 {{no matching function for call to 'g'}}
                    // expected-note@-1 {{in instantiation of function template specialization 'apply<R::X, S::Z>' requested here}}
}
