// RUN: %clang_cc1 -fsyntax-only -verify %s
class G {};
template <Foo> // expected-error{{unknown type name 'Foo'}} \
               // expected-note{{template parameter is declared here}}
class Bar {};

class Bar<G> blah_test; // expected-error{{template argument for non-type template parameter must be an expression}}
