// RUN: %clang_cc1 -fsyntax-only -verify %s

class A {};

namespace B {
  namespace A {} // expected-note{{namespace '::B::A' defined here}} \
                 // expected-note 2{{namespace 'B::A' defined here}}
  using namespace A ;
}

namespace C {} // expected-note{{namespace 'C' defined here}}

namespace D {
  
  class C {
    
    using namespace B ; // expected-error{{not allowed}}
  };
  
  namespace B {}
  
  using namespace C ;
  using namespace B::A ; // expected-error{{no namespace named 'A' in namespace 'D::B'; did you mean '::B::A'?}}
  using namespace ::B::A ;
  using namespace ::D::F ; // expected-error{{expected namespace name}}
  using namespace ::D::C ; // expected-error{{no namespace named 'C' in namespace 'D'; did you mean simply 'C'?}}
}

using namespace ! ; // expected-error{{expected namespace name}}
using namespace A ; // expected-error{{no namespace named 'A'; did you mean 'B::A'?}}
using namespace ::A // expected-error{{no namespace named 'A' in the global namespace; did you mean 'B::A'?}} \
                    // expected-error{{expected ';' after namespace name}}
                    B ; 

void test_nslookup() {
  int B;
  class C;
  using namespace B;
  using namespace C;
}
