// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime-has-weak -x objective-c -fobjc-arc -verify %s

void *memset(void *, int, __SIZE_TYPE__);
void bzero(void *, __SIZE_TYPE__);
void *memcpy(void *, const void *, __SIZE_TYPE__);
void *memmove(void *, const void *, __SIZE_TYPE__);

struct Trivial {
  int f0;
  volatile int f1;
};

struct NonTrivial0 {
  int f0;
  __weak id f1; // expected-note 2 {{non-trivial to default-initialize}} expected-note 2 {{non-trivial to copy}}
  volatile int f2;
  id f3[10]; // expected-note 2 {{non-trivial to default-initialize}} expected-note 2 {{non-trivial to copy}}
};

struct NonTrivial1 {
  id f0; // expected-note 2 {{non-trivial to default-initialize}} expected-note 2 {{non-trivial to copy}}
  int f1;
  struct NonTrivial0 f2;
};

void testTrivial(struct Trivial *d, struct Trivial *s) {
  memset(d, 0, sizeof(struct Trivial));
  bzero(d, sizeof(struct Trivial));
  memcpy(d, s, sizeof(struct Trivial));
  memmove(d, s, sizeof(struct Trivial));
}

void testNonTrivial1(struct NonTrivial1 *d, struct NonTrivial1 *s) {
  memset(d, 0, sizeof(struct NonTrivial1)); // expected-warning {{that is not trivial to primitive-default-initialize}} expected-note {{explicitly cast the pointer to silence}}
  memset((void *)d, 0, sizeof(struct NonTrivial1));
  bzero(d, sizeof(struct NonTrivial1)); // expected-warning {{that is not trivial to primitive-default-initialize}} expected-note {{explicitly cast the pointer to silence}}
  memcpy(d, s, sizeof(struct NonTrivial1)); // expected-warning {{that is not trivial to primitive-copy}} expected-note {{explicitly cast the pointer to silence}}
  memmove(d, s, sizeof(struct NonTrivial1)); // expected-warning {{that is not trivial to primitive-copy}} expected-note {{explicitly cast the pointer to silence}}
}
