// RUN: %clang_analyze_cc1 -w -x c -analyzer-checker=core,unix -analyzer-output=text -verify %s

// Avoid the crash when finding the expression for tracking the origins
// of the null pointer for path notes.
void pr34373() {
  int *a = 0; // expected-note{{'a' initialized to a null pointer value}}
  (a + 0)[0]; // expected-warning{{Array access results in a null pointer dereference}}
              // expected-note@-1{{Array access results in a null pointer dereference}}
}

typedef __typeof(sizeof(int)) size_t;
void *memcpy(void *dest, const void *src, unsigned long count);

void f1(char *source) {
  char *destination = 0; // expected-note{{'destination' initialized to a null pointer value}}
  memcpy(destination + 0, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                       // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}

void f2(char *source) {
  char *destination = 0; // expected-note{{'destination' initialized to a null pointer value}}
  memcpy(destination - 0, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                       // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}

void f3(char *source) {
  char *destination = 0; // expected-note{{'destination' initialized to a null pointer value}}
  destination = destination + 0; // expected-note{{Null pointer value stored to 'destination'}}
  memcpy(destination, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                   // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}

void f4(char *source) {
  char *destination = 0; // expected-note{{'destination' initialized to a null pointer value}}
  destination = destination - 0; // expected-note{{Null pointer value stored to 'destination'}}
  memcpy(destination, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                   // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}

void f5(char *source) {
  char *destination1 = 0; // expected-note{{'destination1' initialized to a null pointer value}}
  char *destination2 = destination1 + 0; // expected-note{{'destination2' initialized to a null pointer value}}
  memcpy(destination2, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                    // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}

void f6(char *source) {
  char *destination1 = 0; // expected-note{{'destination1' initialized to a null pointer value}}
  char *destination2 = destination1 - 0; // expected-note{{'destination2' initialized to a null pointer value}}
  memcpy(destination2, source, 10); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
                                    // expected-note@-1{{Null pointer passed as 1st argument to memory copy function}}
}
