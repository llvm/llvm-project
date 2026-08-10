// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// Ensure that when enforcing access control an unqualified template name with
// explicit template arguments, we don't lose the context of the name lookup
// because of the required early lookup to determine if it names a template.
namespace PR7163 {
  template <typename R, typename P> void h(R (*func)(P)) {}
  class C {
    template <typename T> static void g(T*) {};
   public:
    void f() { h(g<int>); }
  };
}
