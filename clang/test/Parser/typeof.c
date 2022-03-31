// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef int TInt;

static void test(void) {
  int *pi;

  int typeof (int) aIntInt; // expected-error{{cannot combine with previous 'int' declaration specifier}}
  short typeof (int) aShortInt; // expected-error{{'short typeof' is invalid}} 
  int int ttt; // expected-error{{cannot combine with previous 'int' declaration specifier}}
  typeof(TInt) anInt; 
  short TInt eee; // expected-error{{expected ';' at end of declaration}}
  void ary[7] fff; // expected-error{{array has incomplete element type 'void'}} expected-error{{expected ';' at end of declaration}}
  typeof(void ary[7]) anIntError; // expected-error{{expected ')'}} expected-note {{to match this '('}}  expected-error {{variable has incomplete type 'typeof(void)' (aka 'void')}}
  typeof(const int) aci; 
  const typeof (*pi) aConstInt; 
  int xx;
  int *i;
}

// <rdar://problem/8237491>
void test2(void) {
    int a;
    short b;
    __typeof__(a) (*f)(__typeof__(b));    
}
