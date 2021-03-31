// RUN: %clang_cc1 -fsyntax-only -verify %s

void *test1(void) { return 0; }

void test2 (const struct {int a;} *x) {
  // expected-note@-1 {{variable 'x' declared const here}}

  x->a = 10;
  // expected-error-re@-1 {{cannot assign to variable 'x' with const-qualified type 'const struct (unnamed struct at {{.*}}assign.c:5:19) *'}}
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

// PR39946: Recursive checking of hasConstFields caused stack overflow.
struct L { // expected-note {{definition of 'struct L' is not complete until the closing '}'}}
  struct L field; // expected-error {{field has incomplete type 'struct L'}}
};
void testL(struct L *l) {
  *l = 0; // expected-error {{assigning to 'struct L' from incompatible type 'int'}}
}

// Additionally, this example overflowed the stack when figuring out the field.
struct M1; // expected-note {{forward declaration of 'struct M1'}}
struct M2 {
  //expected-note@+1 {{nested data member 'field' declared const here}}
  const struct M1 field; // expected-error {{field has incomplete type 'const struct M1'}}
};
struct M1 {
  struct M2 field;
};

void testM(struct M1 *l) {
  *l = 0; // expected-error {{cannot assign to lvalue with nested const-qualified data member 'field'}}
}

struct N1; // expected-note {{forward declaration of 'struct N1'}}
struct N2 {
  struct N1 field; // expected-error {{field has incomplete type 'struct N1'}}
};
struct N1 {
  struct N2 field;
};

void testN(struct N1 *l) {
  *l = 0; // expected-error {{assigning to 'struct N1' from incompatible type 'int'}}
}
