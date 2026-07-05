// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
    int VA;
    void FA() {}
    struct SA { int V; };
}

using A::VA;
using A::FA;
using typename A::SA;

int main()
{
    VA = 1;
    FA();
    SA x;   //Still needs handling.
}

struct B {
    void f(char){};
    void g(char){};
};
struct D : B {
    using B::f;
    void f(int);
    void g(int);
};
void D::f(int) { f('c'); } // calls B::f(char)
void D::g(int) { g('c'); } // recursively calls D::g(int)

namespace E {
    template <typename TYPE> int funcE(TYPE arg) { return(arg); }
}

using E::funcE<int>; // expected-error{{using declaration cannot refer to a template specialization}}

namespace F {
    struct X;
}

using F::X;
// Should have some errors here.  Waiting for implementation.
void X(int);
struct X *x;


namespace ShadowedTagNotes {

namespace foo {
  class Bar {};
}

void Bar(int); // expected-note{{class 'Bar' is hidden by a non-type declaration of 'Bar' here}}
using foo::Bar;

void ambiguity() {
   const Bar *x; // expected-error{{must use 'class' tag to refer to type 'Bar' in this scope}}
}

} // namespace ShadowedTagNotes
