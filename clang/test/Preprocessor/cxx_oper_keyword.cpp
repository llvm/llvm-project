// RUN: not %clang_cc1 %s -E
// RUN: %clang_cc1 %s -E -fno-operator-names

// Not valid in C++ unless -fno-operator-names is passed.
#define and foo


