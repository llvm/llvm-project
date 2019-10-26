// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
typedef struct { float b; } c;
void *a();
void *d() {
  return a();
}

void no_find_last_store() {
  c *e = d(); // expected-note{{'e' initialized here}}

  (void)(e || e->b); // expected-note{{Assuming 'e' is null}}
      // expected-note@-1{{Left side of '||' is false}}
      // expected-note@-2{{Access to field 'b' results in a dereference of a null pointer (loaded from variable 'e')}}
      // expected-warning@-3{{Access to field 'b' results in a dereference of a null pointer (loaded from variable 'e')}}
}
