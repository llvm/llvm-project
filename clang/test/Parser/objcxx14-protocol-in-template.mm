// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 %s

template<class T> class vector {};
@protocol P @end

// expected-no-diagnostics

template <typename Functor> void F(Functor functor) {}

// Test protocol in template within lambda capture initializer context.
void z() {
  id<P> x = 0;
  (void)x;
  F( [ x = vector<id<P>>{} ] {} );
}
