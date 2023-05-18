// RUN: not %clang_cc1 -fsyntax-only -std=c++11 %s
// PR7481
decltype(;
struct{
  a
}

// PR14549. Must be at end of file.
decltype(
