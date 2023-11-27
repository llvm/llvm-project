// RUN: %clang_cc1 -O2 -verify -emit-codegen-only %s

__attribute__((error("oh no foo"))) void foo(void);

__attribute__((error("oh no bar"))) void bar(void);

int x(void) {
  return 8 % 2 == 1;
}
void baz(void) {
  foo(); // expected-error {{call to 'foo' declared with 'error' attribute: oh no foo}}
         // expected-note@* {{called by function 'baz'}}
  if (x())
    bar();
}

// FIXME: indirect call detection not yet supported.
void (*quux)(void);

void indirect(void) {
  quux = foo;
  quux();
}

static inline void a(int x) {
  if (x == 10)
    foo(); // expected-error {{call to 'foo' declared with 'error' attribute: oh no foo}}
           // expected-note@* {{called by function 'a'}}
           // expected-note@* {{inlined by function 'b'}}
           // expected-note@* {{inlined by function 'd'}}
}

static inline void b() {
  a(10);
}

void c() {
  a(9);
}

void d() {
  b();
}
