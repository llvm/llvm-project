// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface XX
int x;  // expected-error {{cannot declare variable inside @interface or @protocol}}
int one=1;  // expected-error {{cannot declare variable inside @interface or @protocol}}
@end

@protocol PPP
int ddd; // expected-error {{cannot declare variable inside @interface or @protocol}}
@end

@interface XX(CAT)
  char * III; // expected-error {{cannot declare variable inside @interface or @protocol}}
  extern int OK;
@end

@interface XX()
  char * III2; // expected-error {{cannot declare variable inside @interface or @protocol}}
  extern int OK2;
@end


int main( int argc, const char *argv[] ) {
    return x+one;
}

