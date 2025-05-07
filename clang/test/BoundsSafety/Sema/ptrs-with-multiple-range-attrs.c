// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

void foo(int *__counted_by(len) __counted_by(len+2)* buf, int len); // expected-error{{pointer cannot have more than one count attribute}}
void bar(int **__counted_by(len) buf __counted_by(len + 1), int len); // expected-error{{pointer cannot have more than one count attribute}}
int *__counted_by(len) __counted_by(len + 2) baz(int len); // expected-error{{pointer cannot have more than one count attribute}}

int *__counted_by(len) bazx(int len);
int *__counted_by(len) bazx(int len); // ok - same count expr
int *__counted_by(len) bazx(int len) {} // ok - same count expr

int *__counted_by(4) bazz();
int *__counted_by(4) bazz(); // ok - same count expr
int *__counted_by(4) bazz() {} // ok - same count expr

// expected-note@+1{{previous declaration is here}}
int *bazy();
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
int *__counted_by(4) bazy() {} // ok

// expected-note@+2{{previous declaration is here}}
// expected-note@+1{{previous declaration is here}}
int *__counted_by(len) bayz(unsigned len);
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
int *bayz(unsigned len) {}
// expected-error@+1{{conflicting '__sized_by' attribute with the previous function declaration}}
int *__sized_by(len) bayz(unsigned len) {}

// expected-note@+1{{previous declaration is here}}
int *baxz(unsigned len, int **pptr);
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
int *baxz(unsigned len, int *__counted_by(len)* pptr) {} // ok

// expected-note@+1{{previous declaration is here}}
int *baxy(unsigned len, int *__counted_by(len)* pptr);
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
int *baxy(unsigned len, int ** pptr) {}

char *__counted_by(len) qux(int len);
// expected-note@+1{{previous declaration is here}}
char *__sized_by(len) qux(int len);
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
char *__counted_by(4) qux(int len) {}

int *__counted_by(len) __counted_by(len) quux(int len) {} // ok - same count expr

int *__counted_by(len) __counted_by(cnt) quuz(int len, int cnt) {} // expected-error{{pointer cannot have more than one count attribute}}

void corge(int *__counted_by(len) ptr, int len);
void corge(int *__counted_by(len) ptr, int len); // ok
void corge(int *__counted_by(len) ptr, int len) {} // ok

// expected-note@+1{{previous declaration is here}}
int ** grault(int len);
// expected-error@+1{{conflicting '__counted_by' attribute with the previous function declaration}}
int *__counted_by(len) grault(int len) {}

int ** grault2(int len);
// expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
int *__counted_by(len)* grault2(int len) {}

struct S {
    int *__counted_by(l) bp2 __counted_by(l+1); // expected-error{{pointer cannot have more than one count attribute}}
    int l;
};

struct T {
    unsigned n;
    int * end;
    int *__counted_by(n) __ended_by(end) ptr; // expected-error{{pointer cannot have count and range at the same time}}
    int *__ended_by(end) __counted_by(n) ptr2; // expected-error{{pointer cannot have count and range at the same time}}
};

void end1(int *__ended_by(b) a, int *b); // OK
void end1(int *__ended_by(b) a, int *b); // OK redeclaration

// expected-note@+3{{previous declaration is here}} \
   expected-note@+3{{previous declaration is here}} \
   expected-note@+3{{previous declaration is here}}
void end1(int *__ended_by(b) a, int *b) {} // OK definition over declaration

void end1(int *a, int *b); // expected-error{{conflicting '__ended_by' attribute with the previous function declaration}}
void end1(int *a, int *__ended_by(a) b); // expected-error{{conflicting '__ended_by' attribute with the previous function declaration}}
void end1(int *__ended_by(b) a, int *__ended_by(a) b); // expected-error{{conflicting '__ended_by' attribute with the previous function declaration}}

void end2(int *__ended_by(b) a, int *__ended_by(c) b, int *c); // OK
void end2(int *__ended_by(b) a, int *__ended_by(c) b, int *c); // OK

// expected-note@+1{{previous declaration is here}}
void end2(int *__ended_by(b) a, int *__ended_by(c) b, int *c) {} // OK definition over declaration

void end2(int *__ended_by(b) a, int *__ended_by(c) b, int *__ended_by(a) c); // expected-error{{conflicting '__ended_by' attribute with the previous function declaration}}

#ifdef __started_by
// this is not currently compiled; it's meant to break if/when __started_by
// explicitly becomes a thing
void start(int *a, int *__started_by(a) b); // OK
void start(int *a, int *__started_by(a) b); // OK redeclaration
// expected-note@+1{{previous declaration is here}} expected-note@+1{{previous declaration is here}}
void start(int *a, int *__started_by(a) b); // OK definition over declaration

void start(int *a, int *__started_by(a) b); // expected-error{{conflicting '__started_by' attribute with the previous function declaration}}
void start(int *a, int *__started_by(a) b); // expected-error{{conflicting '__started_by' attribute with the previous function declaration}}
#endif
