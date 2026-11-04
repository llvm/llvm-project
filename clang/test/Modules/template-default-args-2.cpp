// RUN: %clang_cc1 -fmodules -verify -fno-modules-error-recovery -fno-spell-checking %s

#pragma clang module build A
module A {
  explicit module X {}
  explicit module Y {}
}
#pragma clang module contents
#pragma clang module begin A.X
namespace N {
  template<class, class = void> struct Foo1 {};
  template<class, class> struct Foo2 {};
}
#pragma clang module end

#pragma clang module begin A.Y
#pragma clang module import A.X
namespace N {
  template<class, class> struct Foo1;
  template<class, class = void> struct Foo2;
}
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import A.X

N::Foo1<int> t1;
N::Foo2<int> t2;
// expected-error@-1 {{default argument of 'Foo2' must be imported from module 'A.Y' before it is required}}
// expected-note@* {{default argument declared here is not reachable}}
