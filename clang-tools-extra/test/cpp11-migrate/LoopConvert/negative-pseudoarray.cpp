// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

// Single FileCheck line to make sure that no loops are converted.
// CHECK-NOT: for ({{.*[^:]:[^:].*}})

const int N = 6;
dependent<int> v;
dependent<int> *pv;

transparent<dependent<int> > cv;
int sum = 0;

// Checks for the index start and end:
void indexStartAndEnd() {
  for (int i = 0; i < v.size() + 1; ++i)
    sum += v[i];

  for (int i = 0; i < v.size() - 1; ++i)
    sum += v[i];

  for (int i = 1; i < v.size(); ++i)
    sum += v[i];

  for (int i = 1; i < v.size(); ++i)
    sum += v[i];

  for (int i = 0; ; ++i)
    sum += (*pv)[i];
}

// Checks for invalid increment steps:
void increment() {
  for (int i = 0; i < v.size(); --i)
    sum += v[i];

  for (int i = 0; i < v.size(); i)
    sum += v[i];

  for (int i = 0; i < v.size();)
    sum += v[i];

  for (int i = 0; i < v.size(); i += 2)
    sum ++;
}

// Checks to make sure that the index isn't used outside of the container:
void indexUse() {
  for (int i = 0; i < v.size(); ++i)
    v[i] += 1 + i;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int badIndex;
  for (int i = 0; badIndex < v.size(); ++i)
    sum += v[i];

  for (int i = 0; i < v.size(); ++badIndex)
    sum += v[i];

  for (int i = 0; badIndex < v.size(); ++badIndex)
    sum += v[i];

  for (int i = 0; badIndex < v.size(); ++badIndex)
    sum += v[badIndex];
}

// Checks for an array indexed in addition to the container.
void multipleArrays() {
  int badArr[N];

  for (int i = 0; i < v.size(); ++i)
    sum += v[i] + badArr[i];

  for (int i = 0; i < v.size(); ++i)
    sum += badArr[i];

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += k + 2;
  }

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += v[i] + k;
  }
}

// Checks for multiple containers being indexed container.
void multipleContainers() {
  dependent<int> badArr;

  for (int i = 0; i < v.size(); ++i)
    sum += v[i] + badArr[i];

  for (int i = 0; i < v.size(); ++i)
    sum += badArr[i];

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += k + 2;
  }

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += v[i] + k;
  }
}

// Check to make sure that dereferenced pointers-to-containers behave nicely
void derefContainer() {
  // Note the dependent<T>::operator*() returns another dependent<T>.
  // This test makes sure that we don't allow an arbitrary number of *'s.
  for (int i = 0; i < pv->size(); ++i)
    sum += (**pv).at(i);

  for (int i = 0; i < pv->size(); ++i)
    sum += (**pv)[i];
}

void wrongEnd() {
  int bad;
  for (int i = 0, e = v.size(); i < bad; ++i)
    sum += v[i];
}
