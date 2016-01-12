// This test checks that intersecting ranges does not cause 'system is over constrained' assertions in the case of eg: 32 bits unsigned integers getting their range from 64 bits signed integers.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-store=region -verify %s

void clang_analyzer_warnIfReached();

void f1(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index + 1 == 0) // because of foo range, index is in range [0; UINT_MAX]
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f2(unsigned long foo)
{
  int index = -1;
  if (index < foo) index = foo; // index equals ULONG_MAX
  if (index + 1 == 0)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // no-warning
}

void f3(unsigned long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index + 1 == 0)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f4(long foo)
{
  int index = -1;
  if (index < foo) index = foo;
  if (index + 1 == 0)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f5(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index == -1)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f6(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index == -1)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f7(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index - 1 == 0) // Was not reached prior fix.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f8(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index + 1L == 0L)
    clang_analyzer_warnIfReached(); // no-warning
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f9(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index - 1L == 0L) // Was not reached prior fix.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f10(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index + 1 == 0L)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f11(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index + 1UL == 0L)
    clang_analyzer_warnIfReached(); // no-warning
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f12(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  if (index - 1UL == 0L) // Was not reached prior fix.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f13(int foo)
{
  unsigned short index = -1;
  if (index < foo) index = foo;
  if (index + 1 == 0)
    clang_analyzer_warnIfReached(); // no-warning
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f14(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  long bar = foo;
  if (index + 1 == 0)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void f15(long foo)
{
  unsigned index = -1;
  if (index < foo) index = foo;
  unsigned int tmp = index + 1;
  if (tmp == 0)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  else
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
