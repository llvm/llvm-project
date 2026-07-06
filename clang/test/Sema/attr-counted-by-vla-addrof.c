/* TO_UPSTREAM(BoundsSafety) ON */
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-late-parse-attributes -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -verify=bounds %s

// expected-no-diagnostics

#define __counted_by(f) __attribute__((counted_by(f)))

struct annotated {
  int count;
  char array[] __counted_by(count);
};

void test1(struct annotated *ptr) {
  // bounds-note@+1{{remove '&' to get address as 'char *' instead of 'char (*)[] __counted_by(count)' (aka 'char (*)[]')}}
  (void)&ptr->array; // bounds-error{{cannot take address of incomplete __counted_by array}}
}

void test2(struct annotated *ptr) {
  // bounds-note@+1{{remove '&' to get address as 'char *' instead of 'char (*)[] __counted_by(count)' (aka 'char (*)[]')}}
  (void)&*&*&*&ptr->array; // bounds-error{{cannot take address of incomplete __counted_by array}}
}
/* TO_UPSTREAM(BoundsSafety) OFF */
