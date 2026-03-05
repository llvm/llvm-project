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
  __weak id f1;// #NonTrivial0_f1
  volatile int f2;
  id f3[10]; // expected-note 2 {{non-trivial to default-initialize}} expected-note 2 {{non-trivial to copy}}
};

struct NonTrivial1 {
  id f0; // #NonTrivial1_f0
  int f1;
  struct NonTrivial0 f2;
};

void testNonTrivial0(struct NonTrivial1 *d, struct NonTrivial1 *s, int i) {
  memset(d, 0, sizeof(struct NonTrivial1));
}
void testTrivial(struct Trivial *d, struct Trivial *s, int i) {
  memset(d, 0, sizeof(struct Trivial));
  memset(d, 1, sizeof(struct Trivial));
  memset(d, i, sizeof(struct Trivial));
  bzero(d, sizeof(struct Trivial));
  memcpy(d, s, sizeof(struct Trivial));
  memmove(d, s, sizeof(struct Trivial));
}

void testNonTrivial1(struct NonTrivial1 *d, struct NonTrivial1 *s, int i) {
  memset(d, 0, sizeof(struct NonTrivial1));
  memset(d, 1, sizeof(struct NonTrivial1)); // #memset_d_1
  // expected-warning@#memset_d_1 {{that is not trivial to primitive-default-initialize}}
  // expected-note@#NonTrivial1_f0 {{non-trivial to default-initialize}}
  // expected-note@#NonTrivial0_f1 {{non-trivial to default-initialize}}
  // expected-note@#memset_d_1 {{explicitly cast the pointer to silence}}
  memset(d, i, sizeof(struct NonTrivial1)); // #memset_d_i
  // expected-warning@#memset_d_i {{that is not trivial to primitive-default-initialize}}
  // expected-note@#NonTrivial1_f0 {{non-trivial to default-initialize}}
  // expected-note@#NonTrivial0_f1 {{non-trivial to default-initialize}}
  // expected-note@#memset_d_i {{explicitly cast the pointer to silence}}
  memset((void *)d, 0, sizeof(struct NonTrivial1));
  bzero(d, sizeof(struct NonTrivial1));
  memcpy(d, s, sizeof(struct NonTrivial1)); // #memcpy_d
  // expected-warning@#memcpy_d {{that is not trivial to primitive-copy}}
  // expected-note@#NonTrivial1_f0 {{non-trivial to copy}}
  // expected-note@#NonTrivial0_f1 {{non-trivial to copy}}
  // expected-note@#memcpy_d {{explicitly cast the pointer to silence}}
  memmove(d, s, sizeof(struct NonTrivial1)); // #memmove_d
  // expected-warning@#memmove_d {{that is not trivial to primitive-copy}}
  // expected-note@#NonTrivial1_f0 {{non-trivial to copy}}
  // expected-note@#NonTrivial0_f1 {{non-trivial to copy}}
  // expected-note@#memmove_d {{explicitly cast the pointer to silence}}
}
