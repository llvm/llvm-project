// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -fsyntax-only -verify %s

#pragma fenv_access (on)
#pragma fenv_access (off)

#pragma fenv_access // expected-warning{{missing '(' after '#pragma fenv_access'}}
#pragma fenv_access foo // expected-warning{{missing '(' after '#pragma fenv_access'}}
#pragma fenv_access on // expected-warning{{missing '(' after '#pragma fenv_access'}}
#pragma fenv_access ( // expected-warning{{incorrect use of '#pragma fenv_access}}
#pragma fenv_access (on // expected-warning{{missing ')' after '#pragma fenv_access'}}
#pragma fenv_access (on) foo // expected-warning{{extra tokens at end of '#pragma fenv_access'}}

void f() {
  (void)0;
  #pragma fenv_access (on) // expected-error{{'#pragma fenv_access' can only appear at file scope or at the start of a compound statement}}
}
