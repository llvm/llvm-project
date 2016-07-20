// RUN: clang-tidy %s -checks=-*,cppcoreguidelines-pro-bounds-constant-array-index -- -std=c++03 | count 0

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.
template <int index> struct B {
  int get() {
    // The next line used to crash the check (in C++03 mode only).
    return x[index];
  }
  int x[3];
};
