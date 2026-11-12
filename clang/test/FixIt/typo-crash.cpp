// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR10355
template<typename T> void template_id1() {
  template_id2<> t; // expected-error-re {{no template named 'template_id2'{{$}}}}
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
