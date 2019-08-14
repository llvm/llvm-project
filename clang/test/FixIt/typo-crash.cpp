// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: The diagnostics and recovery here are very, very poor.

// PR10355
template<typename T> void template_id1() { // expected-note {{'template_id1' declared here}} \
  // expected-note {{possible target for call}}
  template_id2<> t; // expected-error {{no template named 'template_id2'; did you mean 'template_id1'?}} \
  // expected-error {{expected ';' after expression}} \
  // expected-error {{reference to overloaded function could not be resolved; did you mean to call it?}} \
  // expected-error {{use of undeclared identifier 't'}}
 }

// FIXME: It would be nice if we could get this correction right.
namespace PR12297 {
  namespace A {
    typedef short   T;
    
    namespace B {
      typedef short   T;
        
      T global(); // expected-note {{'::PR12297::global' declared here}}
    }
  }

  using namespace A::B;

  // FIXME: Adding '::PR12297::' is not needed as removing 'A::' is sufficient
  T A::global(); // expected-error {{out-of-line declaration of 'global' does not match any declaration in namespace 'PR12297::A'; did you mean '::PR12297::global'?}}
}
