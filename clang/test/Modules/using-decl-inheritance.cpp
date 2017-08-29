// RUN: %clang_cc1 -x c++ -fmodules -fmodules-local-submodule-visibility -fmodules-cache-path=%t %s -verify
// RUN: %clang_cc1 -x c++ -fmodules -fmodules-cache-path=%t %s -verify

// expected-no-diagnostics

#pragma clang module build A
  module A { }
#pragma clang module contents
#pragma clang module begin A
struct A {
   virtual void Foo(double x) const;
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
  module B { }
#pragma clang module contents
#pragma clang module begin B
#pragma clang module import A
struct B : A {
   using A::Foo;
   virtual void Foo(double x) const;
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import B

int main() {
  B b;
  b.Foo(1.0);
}

