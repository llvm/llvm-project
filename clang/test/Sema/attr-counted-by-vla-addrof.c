/* TO_UPSTREAM(BoundsSafety) ON */
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-late-parse-attributes %s -verify

#define __counted_by(f) __attribute__((counted_by(f)))
#define __bdos(P) __builtin_dynamic_object_size(P, 0)

typedef long unsigned int size_t;

struct annotated {
  int count;
  char array[] __counted_by(count);
};

size_t test1(struct annotated *ptr) {
  // expected-note@+1{{remove '&' to get address as 'char *' instead of 'char (*)[] __counted_by(count)' (aka 'char (*)[]')}}
  return __bdos(&ptr->array); // expected-error{{cannot take address of incomplete __counted_by array}}
}

size_t test2(struct annotated *ptr) {
  // expected-note@+1{{remove '&' to get address as 'char *' instead of 'char (*)[] __counted_by(count)' (aka 'char (*)[]')}}
  return __bdos(&*&*&*&ptr->array); // expected-error{{cannot take address of incomplete __counted_by array}}
}
/* TO_UPSTREAM(BoundsSafety) OFF */
