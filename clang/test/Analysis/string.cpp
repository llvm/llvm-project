// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -verify %s

// expected-no-diagnostics

// Test functions that are called "memcpy" but aren't the memcpy
// we're looking for. Unfortunately, this test cannot be put into
// a namespace. The out-of-class weird memcpy needs to be recognized
// as a normal C function for the test to make sense.
typedef __typeof(sizeof(int)) size_t;
void *memcpy(void *, const void *, size_t);

struct S {
  static S s1, s2;

  // A weird overload within the class that accepts a structure reference
  // instead of a pointer.
  void memcpy(void *, const S &, size_t);
  void test_in_class_weird_memcpy() {
    memcpy(this, s2, 1); // no-crash
  }
};

// A similarly weird overload outside of the class.
void *memcpy(void *, const S &, size_t);

void test_out_of_class_weird_memcpy() {
  memcpy(&S::s1, S::s2, 1); // no-crash
}
