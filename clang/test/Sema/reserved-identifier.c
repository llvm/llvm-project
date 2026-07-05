// RUN: %clang_cc1 -isystem %S/Inputs -fsyntax-only -verify -Wreserved-identifier -Wno-visibility %s

#include <reserved-identifier.h>

__I_AM_A_SYSTEM_MACRO() // no-warning

void test_system_macro_expansion(void) {
  SOME_SYSTEM_MACRO(); // no-warning
}

#define __oof foo__ // expected-warning {{macro name is a reserved identifier}}

int foo__bar(void) { return 0; }    // no-warning
static int _bar(void) { return 0; } // expected-warning {{identifier '_bar' is reserved because it starts with '_' at global scope}}
static int _Bar(void) { return 0; } // expected-warning {{identifier '_Bar' is reserved because it starts with '_' followed by a capital letter}}
int _foo(void) { return 0; }        // expected-warning {{identifier '_foo' is reserved because it starts with '_' at global scope}}

// This one is explicitly skipped by -Wreserved-identifier
void *_; // no-warning

void foo(unsigned int _Reserved) { // expected-warning {{identifier '_Reserved' is reserved because it starts with '_' followed by a capital letter}}
  unsigned int __1 =               // expected-warning {{identifier '__1' is reserved because it starts with '__'}}
      _Reserved;                   // no-warning
  goto __reserved;                 // expected-warning {{identifier '__reserved' is reserved because it starts with '__'}}
__reserved: // expected-warning {{identifier '__reserved' is reserved because it starts with '__'}}
            ;
  goto _not_reserved;
_not_reserved: ;
}

void foot(unsigned int _not_reserved) {} // no-warning

enum __menu { // expected-warning {{identifier '__menu' is reserved because it starts with '__'}}
  __some,     // expected-warning {{identifier '__some' is reserved because it starts with '__'}}
  _Other,     // expected-warning {{identifier '_Other' is reserved because it starts with '_' followed by a capital letter}}
  _other      // expected-warning {{identifier '_other' is reserved because it starts with '_' at global scope}}
};

struct __babar { // expected-warning {{identifier '__babar' is reserved because it starts with '__'}}
};

struct _Zebulon;   // expected-warning {{identifier '_Zebulon' is reserved because it starts with '_' followed by a capital letter}}
struct _Zebulon2 { // expected-warning {{identifier '_Zebulon2' is reserved because it starts with '_' followed by a capital letter}}
} * p;
struct _Zebulon3 *pp; // expected-warning {{identifier '_Zebulon3' is reserved because it starts with '_' followed by a capital letter}}

typedef struct {
  int _Field; // expected-warning {{identifier '_Field' is reserved because it starts with '_' followed by a capital letter}}
  int _field; // no-warning
} _Typedef;   // expected-warning {{identifier '_Typedef' is reserved because it starts with '_' followed by a capital letter}}

int foobar(void) {
  return foo__bar(); // no-warning
}

struct _reserved { // expected-warning {{identifier '_reserved' is reserved because it starts with '_' at global scope}}
  int a;
} cunf(void) {
  return (struct _reserved){1};
}

// FIXME: According to clang declaration context layering, _preserved belongs to
// the translation unit, so we emit a warning. It's unclear that's what the
// standard mandate, but it's such a corner case we can live with it.
void func(struct _preserved { int a; } r) {} // expected-warning {{identifier '_preserved' is reserved because it starts with '_' at global scope}}

extern char *_strdup(const char *); // expected-warning {{identifier '_strdup' is reserved because it starts with '_' at global scope}}

// Don't warn on redeclaration
extern char *_strdup(const char *); // no-warning

void ok(void) {
  void _ko(void);           // expected-warning {{identifier '_ko' is reserved because it starts with '_' at global scope}}
  extern int _ko_again; // expected-warning {{identifier '_ko_again' is reserved because it starts with '_' at global scope}}
}
