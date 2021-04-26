// RUN: %clang_cc1 %s -verify -fsyntax-only

int good __attribute__((loader_uninitialized));
static int local_ok __attribute__((loader_uninitialized));
int hidden_ok __attribute__((visibility("hidden"))) __attribute__((loader_uninitialized));

const int still_cant_be_const __attribute__((loader_uninitialized));
extern int external_rejected __attribute__((loader_uninitialized));
// expected-error@-1 {{variable 'external_rejected' cannot be declared both 'extern' and with the 'loader_uninitialized' attribute}}

struct S;
extern S incomplete_external_rejected __attribute__((loader_uninitialized));
// expected-error@-1 {{variable 'incomplete_external_rejected' cannot be declared both 'extern' and with the 'loader_uninitialized' attribute}}

int noargs __attribute__((loader_uninitialized(0)));
// expected-error@-1 {{'loader_uninitialized' attribute takes no arguments}}

int init_rejected __attribute__((loader_uninitialized)) = 42;
// expected-error@-1 {{variable with 'loader_uninitialized' attribute cannot have an initializer}}

void func() __attribute__((loader_uninitialized))
// expected-warning@-1 {{'loader_uninitialized' attribute only applies to global variables}}
{
  int local __attribute__((loader_uninitialized));
  // expected-warning@-1 {{'loader_uninitialized' attribute only applies to global variables}}

  static int sl __attribute__((loader_uninitialized));
}

struct s {
  __attribute__((loader_uninitialized)) int field;
  // expected-warning@-1 {{'loader_uninitialized' attribute only applies to global variables}}

  static __attribute__((loader_uninitialized)) int sfield;

} __attribute__((loader_uninitialized));
// expected-warning@-1 {{'loader_uninitialized' attribute only applies to global variables}}

int redef_attr_first __attribute__((loader_uninitialized));
int redef_attr_first;
// expected-error@-1 {{redefinition of 'redef_attr_first'}}
// expected-note@-3 {{previous definition is here}}

int redef_attr_second;
int redef_attr_second __attribute__((loader_uninitialized));
// expected-warning@-1 {{attribute declaration must precede definition}}
// expected-note@-3 {{previous definition is here}}
// expected-error@-3 {{redefinition of 'redef_attr_second'}}
// expected-note@-5 {{previous definition is here}}

struct trivial {};

trivial default_ok __attribute__((loader_uninitialized));
trivial value_rejected  __attribute__((loader_uninitialized)) {};
// expected-error@-1 {{variable with 'loader_uninitialized' attribute cannot have an initializer}}

struct nontrivial
{
  nontrivial() {}
};

nontrivial needs_trivial_ctor __attribute__((loader_uninitialized));
// expected-error@-1 {{variable with 'loader_uninitialized' attribute must have a trivial default constructor}}

struct Incomplete;
Incomplete incomplete __attribute__((loader_uninitialized));
// expected-error@-1 {{variable has incomplete type 'Incomplete'}}
// expected-note@-3 {{forward declaration of 'Incomplete'}}

struct Incomplete s_incomplete __attribute__((loader_uninitialized));
// expected-error@-1 {{variable has incomplete type 'struct Incomplete'}}
// expected-note@-7 {{forward declaration of 'Incomplete'}}
