namespace A {
  namespace B {  // expected-note{{namespace 'A::B' begins here}}
    #include "misplaced-b.h"  // expected-error{{import of module 'Misplaced.Sub_B' appears within namespace 'A::B'}}
  }
}
