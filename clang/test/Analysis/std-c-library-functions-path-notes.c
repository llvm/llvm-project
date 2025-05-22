// RUN: %clang_analyze_cc1 -verify %s \
// RUN:     -analyzer-checker=core,unix.StdCLibraryFunctions \
// RUN:     -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:     -analyzer-output=text

#include "Inputs/std-c-library-functions-POSIX.h"
#define NULL ((void *)0)

char *getenv(const char *);
int isalpha(int);
int isdigit(int);
int islower(int);

char test_getenv() {
  char *env = getenv("VAR"); // \
  // expected-note{{Assuming the environment variable does not exist}} \
  // expected-note{{'env' initialized here}}

  return env[0]; // \
  // expected-warning{{Array access (from variable 'env') results in a null pointer dereference}} \
  // expected-note   {{Array access (from variable 'env') results in a null pointer dereference}}
}

int test_isalpha(int *x, char c, char d) {
  int iad = isalpha(d);
  if (isalpha(c)) {// \
    // expected-note{{Taking true branch}}
    x = NULL; // \
    // expected-note{{Null pointer value stored to 'x'}}
  }

  return *x; // \
  // expected-warning{{Dereference of null pointer (loaded from variable 'x')}} \
  // expected-note   {{Dereference of null pointer (loaded from variable 'x')}}
}

int test_isdigit(int *x, char c) {
  if (!isdigit(c)) {// \
  // expected-note{{Taking true branch}}
    x = NULL; // \
    // expected-note{{Null pointer value stored to 'x'}}
  }

  return *x; // \
  // expected-warning{{Dereference of null pointer (loaded from variable 'x')}} \
  // expected-note   {{Dereference of null pointer (loaded from variable 'x')}}
}

int test_islower(int *x) {
  char c = 'c';
  // No "Assuming..." note. We aren't assuming anything. We *know*.
  if (islower(c)) { // \
    // expected-note{{Taking true branch}}
    x = NULL; // \
    // expected-note{{Null pointer value stored to 'x'}}
  }

  return *x; // \
  // expected-warning{{Dereference of null pointer (loaded from variable 'x')}} \
  // expected-note   {{Dereference of null pointer (loaded from variable 'x')}}
}

int test_bugpath_notes(FILE *f1, char c, FILE *f2) {
  // This test has the purpose of checking that notes appear at correct place.
  long a = ftell(f2); // no note
  if (a == -1) // \
    // expected-note{{Taking false branch}}
    return -1;
  int l = islower(c); // no note
  a = ftell(f1); // \
  // expected-note{{Value assigned to 'a'}} \
  // expected-note{{Assuming that 'ftell' fails}}
  return dup(a); // \
  // expected-warning{{The 1st argument to 'dup' is -1 but should be >= 0}} \
  // expected-note{{The 1st argument to 'dup' is -1 but should be >= 0}}
}

int test_fileno_arg_note(FILE *f1) {
  return dup(fileno(f1)); // no warning
}

int test_readlink_bufsize_zero(char *Buf, size_t Bufsize) {
  ssize_t Ret = readlink("path", Buf, Bufsize); // \
  // expected-note{{Assuming that argument 'bufsize' is 0}} \
  // expected-note{{'Ret' initialized here}}
  return 1 / Ret; // \
  // expected-warning{{Division by zero}} \
  // expected-note{{Division by zero}}
}
