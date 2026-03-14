// RUN: cp %s %t
// RUN: %clang_cc1 -pedantic -fixit -x c++ %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -Werror -x c++ %t
// XFAIL: *

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

struct  S {
	int i;
};

int foo(int S::* ps, S s, S* p)
{
  p.*ps = 1;
  return s->*ps;
}

void foo1(int (S::*ps)(), S s, S* p)
{
  (p.*ps)();
  (s->*ps)();
}

