// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

// Single FileCheck line to make sure that no loops are converted.
// CHECK-NOT: for ({{.*[^:]:[^:].*}})

const int N = 6;
dependent<int> v;
dependent<int> *pv;

int sum = 0;

// Checks to see that non-const member functions are not called on the container
// object.
// These could be conceivably allowed with a lower required confidence level.
void memberFunctionCalled() {
  for (int i = 0; i < v.size(); ++i) {
    sum += v[i];
    v.foo();
  }

  for (int i = 0; i < v.size(); ++i) {
    sum += v[i];
    dependent<int>::iterator it = v.begin();
  }
}
