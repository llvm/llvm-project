// RUN: %clang_cc1 -fsyntax-only -verify %s

void *test1(void) { return 0; }

void test2 (const struct {int a;} *x) {
  // expected-note@-1 {{variable 'x' declared const here}}

  x->a = 10;
  // expected-error-re@-1 {{cannot assign to variable 'x' with const-qualified type 'const struct (anonymous struct at {{.*}}assign.c:5:19) *'}}
}

typedef int arr[10];
void test3() {
  const arr b;      // expected-note {{variable 'b' declared const here}}
  const int b2[10]; // expected-note {{variable 'b2' declared const here}}
  b[4] = 1;         // expected-error {{cannot assign to variable 'b' with const-qualified type 'const arr' (aka 'int const[10]')}}
  b2[4] = 1;        // expected-error {{cannot assign to variable 'b2' with const-qualified type 'const int [10]'}}
}

typedef struct I {
  const int a; // expected-note 4{{nested data member 'a' declared const here}} \
                  expected-note 6{{data member 'a' declared const here}}
} I;
typedef struct J {
  struct I i;
} J;
typedef struct K {
  struct J *j;
} K;

void testI(struct I i1, struct I i2) {
  i1 = i2; // expected-error {{cannot assign to variable 'i1' with const-qualified data member 'a'}}
}
void testJ1(struct J j1, struct J j2) {
  j1 = j2; // expected-error {{cannot assign to variable 'j1' with nested const-qualified data member 'a'}}
}
void testJ2(struct J j, struct I i) {
  j.i = i; // expected-error {{cannot assign to non-static data member 'i' with const-qualified data member 'a'}}
}
void testK1(struct K k, struct J j) {
  *(k.j) = j; // expected-error {{cannot assign to lvalue with nested const-qualified data member 'a'}}
}
void testK2(struct K k, struct I i) {
  k.j->i = i; // expected-error {{cannot assign to non-static data member 'i' with const-qualified data member 'a'}}
}

void testI_(I i1, I i2) {
  i1 = i2; // expected-error {{cannot assign to variable 'i1' with const-qualified data member 'a'}}
}
void testJ1_(J j1, J j2) {
  j1 = j2; // expected-error {{cannot assign to variable 'j1' with nested const-qualified data member 'a'}}
}
void testJ2_(J j, I i) {
  j.i = i; // expected-error {{cannot assign to non-static data member 'i' with const-qualified data member 'a'}}
}
void testK1_(K k, J j) {
  *(k.j) = j; // expected-error {{cannot assign to lvalue with nested const-qualified data member 'a'}}
}
void testK2_(K k, I i) {
  k.j->i = i; // expected-error {{cannot assign to non-static data member 'i' with const-qualified data member 'a'}}
}
