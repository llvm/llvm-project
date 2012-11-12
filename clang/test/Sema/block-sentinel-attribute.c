// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s

void (^e) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (1,1)));

int main() {
  void (^bbad) (int arg, const char * format) __attribute__ ((__sentinel__)) ; // expected-warning {{'sentinel' attribute only supported for variadic blocks}}
  bbad = ^void (int arg, const char * format) __attribute__ ((__sentinel__)) {} ; // expected-warning {{'sentinel' attribute only supported for variadic blocks}}
  void (^b) (int arg, const char * format, ...) __attribute__ ((__sentinel__)) =  // expected-note {{block has been explicitly marked sentinel here}}
    ^ __attribute__ ((__sentinel__)) (int arg, const char * format, ...) {};
  void (^z) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (2))) = ^ __attribute__ ((__sentinel__ (2))) (int arg, const char * format, ...) {}; // expected-note {{block has been explicitly marked sentinel here}}


  void (^y) (int arg, const char * format, ...) __attribute__ ((__sentinel__ (5))) = ^ __attribute__ ((__sentinel__ (5))) (int arg, const char * format, ...) {}; // expected-note {{block has been explicitly marked sentinel here}}

  b(1, "%s", (void*)0); // OK
  b(1, "%s", 0);  // expected-warning {{missing sentinel in block call}}
  z(1, "%s",4 ,1,0);  // expected-warning {{missing sentinel in block call}}
  z(1, "%s", (void*)0, 1, 0); // OK

  y(1, "%s", 1,2,3,4,5,6,7);  // expected-warning {{missing sentinel in block call}}

  y(1, "%s", (void*)0,3,4,5,6,7); // OK

}

