// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

extern "C" int memcmp(const void *s1, const void *s2, decltype(sizeof(0)) n);

struct Padded { char tag; int x; };
struct Dense { int a, b; };

class Poly {
public:
  virtual ~Poly();
  int x;
};

class MixedAccess {
public:
  int a;
private:
  int b;

public:
  int sum() const { return a + b; }
};

void test_basic(Padded *a, Padded *b) {
  memcmp(a, b, sizeof(Padded)); // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'Padded' which does not have a unique object representation; consider comparing the members of the object manually}} \
                                // expected-note{{explicitly cast the pointer to silence this warning}}
}

void test_dense(Dense *a, Dense *b) {
  memcmp(a, b, sizeof(Dense)); // no warning
}

// Dynamic classes are owned by -Wdynamic-class-memaccess; the new warning
// must not fire on top of it.
void test_poly(Poly *a, Poly *b) {
  memcmp(a, b, sizeof(Poly)); // expected-warning{{first operand of this 'memcmp' call is a pointer to dynamic class 'Poly'; vtable pointer will be compared}} \
                              // expected-note{{explicitly cast the pointer to silence this warning}}
}

// Deliberate scope cut: non-standard-layout without padding stays silent here
// (clang-tidy's bugprone-suspicious-memory-comparison still diagnoses it).
void test_mixed(MixedAccess *a, MixedAccess *b) {
  memcmp(a, b, sizeof(MixedAccess)); // no warning
}

template <typename T>
bool eq(T &a, T &b) {
  return memcmp(&a, &b, sizeof(T)) == 0; // expected-warning{{first operand of this 'memcmp' call is a pointer to type 'Padded' which does not have a unique object representation; consider comparing the members of the object manually}} \
                                         // expected-note{{explicitly cast the pointer to silence this warning}}
}

// Dependent length: must not crash, and must respect the size rule once
// instantiated.
template <int N>
bool eqn(Padded &a, Padded &b) {
  return memcmp(&a, &b, N) == 0; // no warning for N < sizeof(Padded)
}

bool test_templates(Padded p1, Padded p2, Dense d1, Dense d2) {
  return eq(p1, p2) && // expected-note{{in instantiation of function template specialization 'eq<Padded>' requested here}}
         eq(d1, d2) && eqn<4>(p1, p2);
}
