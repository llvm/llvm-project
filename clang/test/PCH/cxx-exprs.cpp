// Test this without pch.
// RUN: %clang_cc1 -include %s -verify -std=c++11 %s

// Test with pch.
// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -verify -std=c++11 %s 

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template<typename T>
class New {
  New(const New&);

public:
  New *clone() {
    return new New(*this);
  }
};

#else

New<int> *clone_new(New<int> *n) {
  return n->clone();
}

#endif
