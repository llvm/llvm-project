// RUN: %clang_analyze_cc1 -w -x c -analyzer-checker=core -analyzer-output=text -verify %s

// Avoid the crash when finding the expression for tracking the origins
// of the null pointer for path notes.
void pr34373() {
  int *a = 0; // expected-note{{'a' initialized to a null pointer value}}
  (a + 0)[0]; // expected-warning{{Array access results in a null pointer dereference}}
              // expected-note@-1{{Array access results in a null pointer dereference}}
}
