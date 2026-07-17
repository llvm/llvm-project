// RUN: %check_clang_tidy -std=c++98-or-later %s cppcoreguidelines-pro-bounds-constant-array-index %t

template <int index> struct B {
  int get() {
    // The next line used to crash the check (in C++03 mode only).
    return x[index];
    // CHECK-FIXES: return x[index];
  }
  int x[3];
};
