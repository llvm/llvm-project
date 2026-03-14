// RUN: %clang_cc1 -fsyntax-only -verify -ftemplate-depth=5 -ftemplate-backtrace-limit=4 %s
// RUN: %clang -fsyntax-only -Xclang -verify -ftemplate-depth-5 -ftemplate-backtrace-limit=4 %s
// RUN: %clang -fsyntax-only -Xclang -verify -ftemplate-depth=5 -ftemplate-backtrace-limit=4 %s

#ifndef NOEXCEPT

template<typename T> struct X : X<T*> { }; \
// expected-error{{recursive template instantiation exceeded maximum depth of 5}} \
// expected-note 3 {{instantiation of template class}} \
// expected-note {{skipping 2 contexts in backtrace}} \
// expected-note {{use -ftemplate-depth=N to increase recursive template instantiation depth}}

void test() {
  (void)sizeof(X<int>); // expected-note {{instantiation of template class}}
}

#else

// RUN: %clang_cc1 -fsyntax-only -verify -ftemplate-depth=5 -ftemplate-backtrace-limit=4 -std=c++11 -DNOEXCEPT %s

template<typename T> struct S {
  S() noexcept(noexcept(S<S>())); \
// expected-error{{recursive template instantiation exceeded maximum depth of 5}} \
// expected-note 3 {{in instantiation of exception spec}} \
// expected-note {{skipping 2 contexts in backtrace}} \
// expected-note {{use -ftemplate-depth=N to increase recursive template instantiation depth}}
};
S<void> t; // expected-note {{in instantiation of exception spec}}

#endif
