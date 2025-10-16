// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c++17 -I %S/Inputs/not-null-terminated-result

// This test case reproduces the crash when the check tries to evaluate
// a value-dependent expression using EvaluateAsInt() in
// bugprone-not-null-terminated-result, where the src parameter of memcpy is
// value-dependent, but the length is not.

// expected-no-diagnostics

#include "not-null-terminated-result-cxx.h"

template<size_t N>
class ValueDependentClass {
public:
  void copyData(char* Dst) {
    const char* Src = reinterpret_cast<const char*>(this);
    // The length parameter is arbitrary, but the crash is not reproduced if it is N.
    memcpy(Dst, Src, 32);
  }
};

template class ValueDependentClass<42>; // The template parameter value is arbitrary.
