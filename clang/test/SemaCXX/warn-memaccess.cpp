// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wnontrivial-memaccess %s

extern "C" void *bzero(void *, unsigned);
extern "C" void *memset(void *, int, unsigned);
extern "C" void *memmove(void *s1, const void *s2, unsigned n);
extern "C" void *memcpy(void *s1, const void *s2, unsigned n);

class TriviallyCopyable {};
class NonTriviallyCopyable { NonTriviallyCopyable(const NonTriviallyCopyable&);};

void test_bzero(TriviallyCopyable* tc,
                 NonTriviallyCopyable *ntc) {
  // OK
  bzero(tc, sizeof(*tc));

  // expected-warning@+2{{first argument in call to 'bzero' is a pointer to non-trivially copyable type 'NonTriviallyCopyable'}}
  // expected-note@+1{{explicitly cast the pointer to silence this warning}}
  bzero(ntc, sizeof(*ntc));

  // OK
  bzero((void*)ntc, sizeof(*ntc));
}

void test_memset(TriviallyCopyable* tc,
                 NonTriviallyCopyable *ntc) {
  // OK
  memset(tc, 0, sizeof(*tc));

  // expected-warning@+2{{first argument in call to 'memset' is a pointer to non-trivially copyable type 'NonTriviallyCopyable'}}
  // expected-note@+1{{explicitly cast the pointer to silence this warning}}
  memset(ntc, 0, sizeof(*ntc));

  // OK
  memset((void*)ntc, 0, sizeof(*ntc));
}


void test_memcpy(TriviallyCopyable* tc0, TriviallyCopyable* tc1,
                 NonTriviallyCopyable *ntc0, NonTriviallyCopyable *ntc1) {
  // OK
  memcpy(tc0, tc1, sizeof(*tc0));

  // expected-warning@+2{{first argument in call to 'memcpy' is a pointer to non-trivially copyable type 'NonTriviallyCopyable'}}
  // expected-note@+1{{explicitly cast the pointer to silence this warning}}
  memcpy(ntc0, ntc1, sizeof(*ntc0));

  // ~ OK
  memcpy((void*)ntc0, ntc1, sizeof(*ntc0));

  // OK
  memcpy((void*)ntc0, (void*)ntc1, sizeof(*ntc0));
}

void test_memmove(TriviallyCopyable* tc0, TriviallyCopyable* tc1,
                 NonTriviallyCopyable *ntc0, NonTriviallyCopyable *ntc1) {
  // OK
  memmove(tc0, tc1, sizeof(*tc0));

  // expected-warning@+2{{first argument in call to 'memmove' is a pointer to non-trivially copyable type 'NonTriviallyCopyable'}}
  // expected-note@+1{{explicitly cast the pointer to silence this warning}}
  memmove(ntc0, ntc1, sizeof(*ntc0));

  // ~ OK
  memmove((void*)ntc0, ntc1, sizeof(*ntc0));

  // OK
  memmove((void*)ntc0, (void*)ntc1, sizeof(*ntc0));
}
