
// RUN: %clang_cc1 -x c -verify -verify=c %s
// RUN: %clang_cc1 -x c++ -verify=expected -verify=cxx %s
// RUN: %clang_cc1 -x objective-c -verify -verify=c %s
// RUN: %clang_cc1 -x objective-c++ -verify=expected -verify=cxx %s

int *__attribute__((bidi_indexable)) gbi; // expected-warning{{attribute ignored}}
int *__attribute__((unsafe_indexable)) gus; // expected-warning{{attribute ignored}}

struct S1 {
  // FIXME: c++ drops counted_by entirely because of late parsing but that's okay for now.
  int *__attribute__((counted_by(len))) ptr; // expected-error{{undeclared identifier 'len'}}
  int len;
};

struct S2 {
  int len;
  // cxx-error@+1{{invalid use of non-static data member 'len'}}
  int *__attribute__((counted_by(len))) ptr; // ok
};

int foo1(int *__attribute__((counted_by(len))) ptr, int len) { // expected-error{{use of undeclared identifier 'len'}}
  int i;
  // cxx-error@+2{{cannot initialize a variable of type}}
  // c-error@+1{{incompatible integer to pointer conversion initializing}}
  int *__attribute__((indexable)) pi = i; // expected-warning{{attribute ignored}}
  int *__attribute__((single)) ps; // expected-warning{{attribute ignored}}
  return 0;
}

// cxx-warning@+1{{'counted_by' attribute ignored}}
int foo2(int len, int *__attribute__((counted_by(len))) ptr); // c-error{{counted_by attribute only applies to non-static data members}}

// c-error@+2{{sized_by attribute only applies to non-static data members}}
// cxx-warning@+1{{'sized_by' attribute ignored}}
void bar1(int size, void *__attribute__((sized_by(size))) ptr);
void bar2(void *__attribute__((sized_by(size))) ptr, int size); // expected-error{{use of undeclared identifier 'size'}}

void baz1(void *end, void *__attribute__((ended_by(end))) start); // expected-warning{{'ended_by' attribute ignored}}
void baz2(void *__attribute__((ended_by(end))) start, void *end); // expected-error{{use of undeclared identifier 'end'}}
