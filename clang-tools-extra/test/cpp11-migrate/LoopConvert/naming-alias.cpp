// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

const int N = 10;

Val Arr[N];
Val &func(Val &);
void sideEffect(int);

void aliasing() {
  // If the loop container is only used for a declaration of a temporary
  // variable to hold each element, we can name the new variable for the
  // converted range-based loop as the temporary variable's name.

  // In the following case, "t" is used as a temporary variable to hold each
  // element, and thus we consider the name "t" aliased to the loop.
  // The extra blank braces are left as a placeholder for after the variable
  // declaration is deleted.
  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i]; { }
    int y = t.x;
  }
  // CHECK: for (auto & t : Arr)
  // CHECK-NOT: Val &{{[a-z_]+}} =
  // CHECK-NEXT: { }
  // CHECK-NEXT: int y = t.x;

  // The container was not only used to initialize a temporary loop variable for
  // the container's elements, so we do not alias the new loop variable.
  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK: for (auto & elem : Arr)
  // CHECK-NEXT: Val &t = elem;
  // CHECK-NEXT: int y = t.x;
  // CHECK-NEXT: int z = elem.x + t.x;

  for (int i = 0; i < N; ++i) {
    Val t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK: for (auto & elem : Arr)
  // CHECK-NEXT: Val t = elem;
  // CHECK-NEXT: int y = t.x;
  // CHECK-NEXT: int z = elem.x + t.x;

  for (int i = 0; i < N; ++i) {
    Val &t = func(Arr[i]);
    int y = t.x;
  }
  // CHECK: for (auto & elem : Arr)
  // CHECK-NEXT: Val &t = func(elem);
  // CHECK-NEXT: int y = t.x;

  int IntArr[N];
  for (unsigned i = 0; i < N; ++i) {
    if (int alias = IntArr[i]) {
      sideEffect(alias);
    }
  }
  // CHECK: for (auto alias : IntArr)
  // CHECK-NEXT: if (alias) {

  for (unsigned i = 0; i < N; ++i) {
    while (int alias = IntArr[i]) {
      sideEffect(alias);
    }
  }
  // CHECK: for (auto alias : IntArr)
  // CHECK-NEXT: while (alias) {

  for (unsigned i = 0; i < N; ++i) {
    switch (int alias = IntArr[i]) {
    default:
      sideEffect(alias);
    }
  }
  // CHECK: for (auto alias : IntArr)
  // CHECK-NEXT: switch (alias) {

  for (unsigned i = 0; i < N; ++i) {
    for (int alias = IntArr[i]; alias < N; ++alias) {
      sideEffect(alias);
    }
  }
  // CHECK: for (auto alias : IntArr)
  // CHECK-NEXT: for (; alias < N; ++alias) {

  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; int alias = IntArr[i]; ++j) {
      sideEffect(alias);
    }
  }
  // CHECK: for (auto alias : IntArr)
  // CHECK-NEXT: for (unsigned j = 0; alias; ++j) {
}

void refs_and_vals() {
  // The following tests check that the transform correctly preserves the
  // reference or value qualifiers of the aliased variable. That is, if the
  // variable was declared as a value, the loop variable will be declared as a
  // value and vice versa for references.

  S s;
  const S s_const = s;

  for (S::const_iterator it = s_const.begin(); it != s_const.end(); ++it) {
    MutableVal alias = *it; { }
    alias.x = 0;
  }
  // CHECK: for (auto alias : s_const)
  // CHECK-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-NEXT: { }
  // CHECK-NEXT: alias.x = 0;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    MutableVal alias = *it; { }
    alias.x = 0;
  }
  // CHECK: for (auto alias : s)
  // CHECK-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-NEXT: { }
  // CHECK-NEXT: alias.x = 0;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    MutableVal &alias = *it; { }
    alias.x = 0;
  }
  // CHECK: for (auto & alias : s)
  // CHECK-NOT: MutableVal &{{[a-z_]+}} =
  // CHECK-NEXT: { }
  // CHECK-NEXT: alias.x = 0;
}
