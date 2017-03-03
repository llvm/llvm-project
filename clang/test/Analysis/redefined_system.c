// RUN: %clang_analyze_cc1 -analyzer-checker=osx,unix,core,alpha.security.taint -w -verify %s
// expected-no-diagnostics

// Make sure we don't crash when someone redefines a system function we reason about.

char memmove ();
char malloc();
char system();
char stdin();
char memccpy();
char free();
char strdup();
char atoi();

int foo () {
  return memmove() + malloc() + system() + stdin() + memccpy() + free() + strdup() + atoi();

}
