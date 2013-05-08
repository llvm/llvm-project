// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert -risk=safe %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

// Single FileCheck line to make sure that no loops are converted.
// CHECK-NOT: for ({{.*[^:]:[^:].*}})

S s;
T t;
U u;

void multipleEnd() {
  for (S::iterator i = s.begin(); i != s.end(); ++i)
    MutableVal k = *i;

  for (T::iterator i = t.begin(); i != t.end(); ++i)
    int k = *i;

  for (U::iterator i = u.begin(); i != u.end(); ++i)
    Val k = *i;
}

void f(X);
void f(S);
void f(T);

void complexContainer() {
  X x;
  for (S::iterator i = x.s.begin(), e = x.s.end(); i != e; ++i) {
    f(x);
    MutableVal k = *i;
  }

  for (T::iterator i = x.t.begin(), e = x.t.end(); i != e; ++i) {
    f(x);
    int k = *i;
  }

  for (S::iterator i = x.s.begin(), e = x.s.end(); i != e; ++i) {
    f(x.s);
    MutableVal k = *i;
  }

  for (T::iterator i = x.t.begin(), e = x.t.end(); i != e; ++i) {
    f(x.t);
    int k = *i;
  }

  for (S::iterator i = x.getS().begin(), e = x.getS().end(); i != e; ++i) {
    f(x.getS());
    MutableVal k = *i;
  }

  X exes[5];
  int index = 0;

  for (S::iterator i = exes[index].getS().begin(),
       e = exes[index].getS().end(); i != e; ++i) {
    index++;
    MutableVal k = *i;
  }
}
