// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -Wnontrivial-memcall %s

extern "C" void *memcpy(void *s1, const void *s2, unsigned n);

class TriviallyCopyable {};
class NonTriviallyCopyable { NonTriviallyCopyable(const NonTriviallyCopyable&);};
struct Incomplete;

void test_memcpy(TriviallyCopyable* tc0, TriviallyCopyable* tc1,
                 NonTriviallyCopyable *ntc0, NonTriviallyCopyable *ntc1,
                 Incomplete *i0, Incomplete *i1) {
  // OK
  memcpy(tc0, tc1, sizeof(*tc0));

  // OK
  memcpy(i0, i1, 10);

  // expected-warning@+2{{first argument in call to 'memcpy' is a pointer to non-trivially copyable type 'NonTriviallyCopyable'}}
  // expected-note@+1{{explicitly cast the pointer to silence this warning}}
  memcpy(ntc0, ntc1, sizeof(*ntc0));

  // ~ OK
  memcpy((void*)ntc0, ntc1, sizeof(*ntc0));

  // OK
  memcpy((void*)ntc0, (void*)ntc1, sizeof(*ntc0));
}
