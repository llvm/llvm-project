// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

// Single FileCheck line to make sure that no loops are converted.
// CHECK-NOT: for ({{.*[^:]:[^:].*}})

S s;
T t;
U u;

struct BadBeginEnd : T {
  iterator notBegin();
  iterator notEnd();
};

void notBeginOrEnd() {
  BadBeginEnd Bad;
  for (T::iterator i = Bad.notBegin(), e = Bad.end(); i != e; ++i)
    int k = *i;

  for (T::iterator i = Bad.begin(), e = Bad.notEnd(); i != e; ++i)
    int k = *i;
}

void badLoopShapes() {
  for (T::iterator i = t.begin(), e = t.end(), f = e; i != e; ++i)
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e; )
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end(); ; ++i)
    int k = *i;

  T::iterator outsideI;
  T::iterator outsideE;

  for (; outsideI != outsideE ; ++outsideI)
    int k = *outsideI;
}

void iteratorArrayMix() {
  int lower;
  const int N = 6;
  for (T::iterator i = t.begin(), e = t.end(); lower < N; ++i)
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end(); lower < N; ++lower)
    int k = *i;
}

struct ExtraConstructor : T::iterator {
  ExtraConstructor(T::iterator, int);
  explicit ExtraConstructor(T::iterator);
};

void badConstructor() {
  for (T::iterator i = ExtraConstructor(t.begin(), 0), e = t.end();
       i != e; ++i)
    int k = *i;
  for (T::iterator i = ExtraConstructor(t.begin()), e = t.end(); i != e; ++i)
    int k = *i;
}

void iteratorMemberUsed() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    i.x = *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = i.x + *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = e.x + *i;
}

void iteratorMethodCalled() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    i.insert(3);

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    if (i != i)
      int k = 3;
}

void iteratorOperatorCalled() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = *(++i);

  for (S::iterator i = s.begin(), e = s.end(); i != e; ++i)
    MutableVal k = *(++i);
}

void differentContainers() {
  T other;
  for (T::iterator i = t.begin(), e = other.end(); i != e; ++i)
    int k = *i;

  for (T::iterator i = other.begin(), e = t.end(); i != e; ++i)
    int k = *i;

  S otherS;
  for (S::iterator i = s.begin(), e = otherS.end(); i != e; ++i)
    MutableVal k = *i;

  for (S::iterator i = otherS.begin(), e = s.end(); i != e; ++i)
    MutableVal k = *i;
}

void wrongIterators() {
  T::iterator other;
  for (T::iterator i = t.begin(), e = t.end(); i != other; ++i)
    int k = *i;
}

struct EvilArrow : U {
  // Please, no one ever write code like this.
  U* operator->();
};

void differentMemberAccessTypes() {
  EvilArrow A;
  for (EvilArrow::iterator i = A.begin(), e = A->end(); i != e; ++i)
    Val k = *i;
  for (EvilArrow::iterator i = A->begin(), e = A.end(); i != e; ++i)
    Val k = *i;
}

void f(const T::iterator &it, int);
void f(const T &it, int);
void g(T &it, int);

void iteratorPassedToFunction() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    f(i, *i);
}

// FIXME: Disallow this except for containers passed by value and/or const
// reference. Or maybe this is correct enough for any container?
void containerPassedToFunction() {
//  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
//    f(t, *i);
//  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
//    g(t, *i);
}

// FIXME: These tests can be removed if this tool ever does enough analysis to
// decide that this is a safe transformation.
// Until then, we don't want it applied.
void iteratorDefinedOutside() {
  T::iterator theEnd = t.end();
  for (T::iterator i = t.begin(); i != theEnd; ++i)
    int k = *i;

  T::iterator theBegin = t.begin();
  for (T::iterator e = t.end(); theBegin != e; ++theBegin)
    int k = *theBegin;
}
