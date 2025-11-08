// Ensures that Clang does not crash in C++ mode, when a nested initializer
// is followed by a designated initializer for a union member of that same
// subobject.
// See issue #166327.

// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

auto main(void) -> int {
  struct Point {
    float x;
    float y;
    union {
      int idx;
      char label;
    } extra;
  };

  struct SavePoint {
    struct Point p;
  };

  SavePoint save = {.p = {.x = 3.0, .y = 4.0}, .p.extra.label = 'p'}; // expected-warning {{nested designators are a C99 extension}}
}
