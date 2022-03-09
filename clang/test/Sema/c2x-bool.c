// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify %s

_Static_assert(_Generic(true, _Bool : 1, default: 0));
_Static_assert(_Generic(false, _Bool : 1, default: 0));

_Static_assert(_Generic(true, bool : 1, default: 0));
_Static_assert(_Generic(false, bool : 1, default: 0));

_Static_assert(_Generic(true, bool : true, default: false));
_Static_assert(_Generic(false, bool : true, default: false));

_Static_assert(true == (bool)+1);
_Static_assert(false == (bool)+0);

_Static_assert(_Generic(+true, bool : 0, unsigned int : 0, signed int : 1, default : 0));

struct S {
  bool b : 1;
} s;
_Static_assert(_Generic(+s.b, bool : 0, unsigned int : 0, signed int : 1, default : 0));

static void *f = false; // expected-warning {{to null from a constant boolean expression}}
static int one = true;
static int zero = false;

static void do_work() {
  char *str = "Foo";
  str[false] = 'f';
  str[true] = 'f';

  char c1[true];
  char c2[false];
}

#if true != 1
#error true should be 1 in the preprocessor
#endif

#if false != 0
#error false should be 0 in the preprocessor
#endif
