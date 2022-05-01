// RUN: %clang_analyze_cc1 -verify %s \
// RUN:     -analyzer-checker=core,apiModeling \
// RUN:     -analyzer-output=text

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

int test_isalpha(int *x, char c) {
  if (isalpha(c)) {// \
    // expected-note{{Assuming the character is alphabetical}} \
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
    // expected-note{{Assuming the character is not a digit}} \
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
