// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cp %t.cpp %t.base
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
//
// See PR15589 for why this test fails.
// XFAIL: *

#include "macro_problem.h"
#include "structures.h"

void side_effect(const myns::MyType &T);

void f() {
  TypedefDerefContainer<myns::MyType> container;
  for (TypedefDerefContainer<myns::MyType>::iterator I = container.begin(),
       E = container.end(); I != E; ++I) {
    myns::MyType &alias = *I;
    // CHECK: myns::MyType &ref = *I;
    side_effect(ref);
  }
}
