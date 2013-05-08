// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

void complexContainer() {
  X exes[5];
  int index = 0;

  for (S::iterator i = exes[index].getS().begin(), e = exes[index].getS().end(); i != e; ++i) {
    MutableVal k = *i;
    MutableVal j = *i;
  }
  // CHECK: for (auto & elem : exes[index].getS())
  // CHECK-NEXT: MutableVal k = elem;
  // CHECK-NEXT: MutableVal j = elem;
}

void f() {
  /// begin()/end() - based for loops here:
  T t;
  for (T::iterator it = t.begin(); it != t.end(); ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK: for (auto & elem : t)
  // CHECK-NEXT: printf("I found %d\n", elem);

  T *pt;
  for (T::iterator it = pt->begin(); it != pt->end(); ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK: for (auto & elem : *pt)
  // CHECK-NEXT: printf("I found %d\n", elem);

  S s;
  for (S::const_iterator it = s.begin(); it != s.end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for (auto & elem : s)
  // CHECK-NEXT: printf("s has value %d\n", (elem).x);

  S *ps;
  for (S::const_iterator it = ps->begin(); it != ps->end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for (auto & p : *ps)
  // CHECK-NEXT: printf("s has value %d\n", (p).x);

  for (S::const_iterator it = s.begin(); it != s.end(); ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK: for (auto & elem : s)
  // CHECK-NEXT: printf("s has value %d\n", elem.x);

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    it->x = 3;
  }
  // CHECK: for (auto & elem : s)
  // CHECK-NEXT: elem.x = 3;

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    (*it).x = 3;
  }
  // CHECK: for (auto & elem : s)
  // CHECK-NEXT: (elem).x = 3;

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    it->nonConstFun(4, 5);
  }
  // CHECK: for (auto & elem : s)
  // CHECK-NEXT: elem.nonConstFun(4, 5);

  U u;
  for (U::iterator it = u.begin(); it != u.end(); ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK: for (auto & elem : u)
  // CHECK-NEXT: printf("s has value %d\n", elem.x);

  for (U::iterator it = u.begin(); it != u.end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for (auto & elem : u)
  // CHECK-NEXT: printf("s has value %d\n", (elem).x);

  U::iterator A;
  for (U::iterator i = u.begin(); i != u.end(); ++i)
    int k = A->x + i->x;
  // CHECK: for (auto & elem : u)
  // CHECK-NEXT: int k = A->x + elem.x;

  dependent<int> v;
  for (dependent<int>::const_iterator it = v.begin();
       it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);

  for (dependent<int>::const_iterator it(v.begin());
       it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);

  doublyDependent<int,int> intmap;
  for (doublyDependent<int,int>::iterator it = intmap.begin();
       it != intmap.end(); ++it) {
    printf("intmap[%d] = %d", it->first, it->second);
  }
  // CHECK: for (auto & elem : intmap)
  // CHECK-NEXT: printf("intmap[%d] = %d", elem.first, elem.second);
}
