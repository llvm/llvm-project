// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=both %s
// RUN: %clang_cc1 -fsyntax-only -std=c17 -verify=both,c17 %s

/* WG14 N3037:
 * Improved tag compatibility
 *
 * Identical tag types have always been compatible across TU boundaries. This
 * paper made identical tag types compatible within the same TU.
 */

struct foo { int a; } p;

void baz(struct foo f); // c17-note {{passing argument to parameter 'f' here}}

void bar(void) {
  struct foo { int a; } q;
  baz(q); // c17-error {{passing 'struct foo' to parameter of incompatible type 'struct foo'}}
}

#define PRODUCT(A ,B) struct prod { A a; B b; }                   // expected-note 2 {{expanded from macro 'PRODUCT'}}
#define SUM(A, B) struct sum { _Bool flag; union { A a; B b; }; } // expected-note 2 {{expanded from macro 'SUM'}}

void func1(PRODUCT(int, SUM(float, double)) x); // both-warning {{declaration of 'struct prod' will not be visible outside of this function}} \
                                                   both-warning {{declaration of 'struct sum' will not be visible outside of this function}} \
                                                   c17-note {{passing argument to parameter 'x' here}}
void func2(PRODUCT(int, SUM(float, double)) y) { // both-warning {{declaration of 'struct prod' will not be visible outside of this function}} \
                                                    both-warning {{declaration of 'struct sum' will not be visible outside of this function}}
  func1(y); // c17-error {{passing 'struct prod' to parameter of incompatible type 'struct prod'}}
}

struct foop { struct { int x; }; }; // c17-note {{previous definition is here}}
struct foop { struct { int x; }; }; // c17-error {{redefinition of 'foop'}}
union barp { int x; float y; };     // c17-note {{previous definition is here}}
union barp { int x; float y; };     // c17-error {{redefinition of 'barp'}}
typedef struct q { int x; } q_t;    // c17-note 2 {{previous definition is here}}
typedef struct q { int x; } q_t;    // c17-error {{redefinition of 'q'}} \
                                       c17-error-re {{typedef redefinition with different types ('struct (unnamed struct at {{.*}})' vs 'struct q')}}
void func3(void) {
  struct S { int x; };       // c17-note {{previous definition is here}}
  struct T { struct S s; };  // c17-note {{previous definition is here}}
  struct S { int x; };       // c17-error {{redefinition of 'S'}}
  struct T { struct S s; };  // c17-error {{redefinition of 'T'}}
}
