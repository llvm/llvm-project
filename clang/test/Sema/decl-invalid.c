// RUN: %clang_cc1 %s -fsyntax-only -verify

// See Sema::ParsedFreeStandingDeclSpec about the double diagnostic
typedef union <anonymous> __mbstate_t;  // expected-error {{declaration of anonymous union must be a definition}} expected-warning {{typedef requires a name}}


// PR2017
void x(void); 
int a(void) {
  int r[x()];  // expected-error {{size of array has non-integer type 'void'}}

  static y ?; // expected-error{{unknown type name 'y'}} \
                 expected-error{{expected identifier or '('}}
}

int; // expected-warning {{declaration does not declare anything}}
typedef int; // expected-warning {{typedef requires a name}}
const int; // expected-warning {{declaration does not declare anything}}
struct; // expected-error {{declaration of anonymous struct must be a definition}} // expected-warning {{declaration does not declare anything}}
typedef int I;
I; // expected-warning {{declaration does not declare anything}}



// rdar://6880449
register int test1;     // expected-error {{illegal storage class on file-scoped variable}}

