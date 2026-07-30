// RUN: %clang_analyze_cc1 -analyzer-checker=osx,unix,core,optin.taint -w -verify %s
// expected-no-diagnostics

// Make sure we don't crash when someone redefines a system function we reason about.

char memmove (void);
char malloc(void);
char system(void);
char stdin(void);
char memccpy(void);
char free(void);
char strdup(void);
char atoi(void);

int foo (void) {
  return memmove() + malloc() + system() + stdin() + memccpy() + free() + strdup() + atoi();

}
