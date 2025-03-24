// RUN: %clang_cc1 -fsyntax-only -std=c23 -verify=both,c23 %s
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

struct food { int (*p)[3]; }; // c23-note {{field 'p' has type 'int (*)[3]' here}} \
                                 c17-note {{previous definition is here}}
struct food { int (*p)[]; };  // c23-error {{type 'struct food' has incompatible definitions}} \
                                 c23-note {{field 'p' has type 'int (*)[]' here}} \
                                 c17-error {{redefinition of 'food'}}
union bard { int x; float y; }; // c23-note {{field has name 'x' here}} \
                                   c17-note {{previous definition is here}}
union bard { int z; float y; }; // c23-error {{type 'union bard' has incompatible definitions}} \
                                   c23-note {{field has name 'z' here}} \
                                   c17-error {{redefinition of 'bard'}}
union purr { int x; float y; }; // c23-note {{field has name 'x' here}} \
                                   c17-note {{previous definition is here}}
union purr { float y; int x; }; // c23-error {{type 'union purr' has incompatible definitions}} \
                                   c23-note {{field has name 'y' here}} \
                                   c17-error {{redefinition of 'purr'}}

// FIXME: this should have a structural equivalence error in C23.
struct [[gnu::packed]] attr_test { // c17-note {{previous definition is here}}
  int x;
};

struct attr_test { // c17-error {{redefinition of 'attr_test'}}
  int x;
};

// FIXME: this should have a structural equivalence error in C23.
struct field_attr_test { // c17-note {{previous definition is here}}
  int x;
  [[gnu::packed]] int y;
};

struct field_attr_test { // c17-error {{redefinition of 'field_attr_test'}}
  int x;
  int y;
};

// Show that equivalent field types are not an issue.
typedef int typedef_of_type_int;
struct equivalent_field_types { // c17-note {{previous definition is here}}
  int x;
};

struct equivalent_field_types { // c17-error {{redefinition of 'equivalent_field_types'}}
  typedef_of_type_int x;
};

struct quals_matter { // c17-note {{previous definition is here}}
  int x;              // c23-note {{field 'x' has type 'int' here}}
};

struct quals_matter { // c17-error {{redefinition of 'quals_matter'}} \
                         c23-error {{type 'struct quals_matter' has incompatible definitions}}                         
  const int x;        // c23-note {{field 'x' has type 'const int' here}}
};

struct qual_order_does_not_matter { // c17-note {{previous definition is here}}
  const volatile int x;
};

struct qual_order_does_not_matter { // c17-error {{redefinition of 'qual_order_does_not_matter'}}
  volatile const int x;
};
