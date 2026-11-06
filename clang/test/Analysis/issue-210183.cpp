// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

// https://github.com/llvm/llvm-project/issues/210183
//
// A pointer member initialized via array-to-pointer decay of a
// reference-to-array constructor parameter used to be modeled as the
// address of the whole array (instead of its first element). Dereferencing
// and storing through that mistyped pointer then reached
// RegionStoreManager::bindArray() with a scalar Init value, crashing on an
// unchecked castAs<nonloc::CompoundVal>().

template <class T> void clang_analyzer_dump(T);

template <class T> struct Span {
  template <int N>
  Span(T (&arr)[N]) : ptr_(arr) {}
  T *data() { return ptr_; }
  T *ptr_;
};

char *ptr();
char buffer[10];

void test() {
  char *p = Span<char>(buffer).data();

  // p and buffer must resolve to the same address: the first element of
  // buffer, not the whole array.
  clang_analyzer_dump(buffer); // expected-warning{{&Element{buffer,0 S64b,char}}}
  clang_analyzer_dump(p);      // expected-warning{{&Element{buffer,0 S64b,char}}}

  int v = (int)(long)ptr();
  *p = v; // no-crash
}
