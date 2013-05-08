// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

#define MAX(a,b) (a > b) ? a : b
#define DEF 5

const int N = 10;
int nums[N];
int sum = 0;

namespace ns {
  struct st {
    int x;
  };
}

void sameNames() {
  int num = 0;
  for (int i = 0; i < N; ++i) {
    printf("Fibonacci number is %d\n", nums[i]);
    sum += nums[i] + 2 + num;
    (void) nums[i];
  }
  // CHECK: for (auto & nums_i : nums)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", nums_i);
  // CHECK-NEXT: sum += nums_i + 2 + num;
  // CHECK-NOT: (void) num;
}

void macroConflict() {
  S MAXs;
  for (S::const_iterator it = MAXs.begin(), e = MAXs.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
    printf("Max of 3 and 5: %d\n", MAX(3,5));
  }
  // CHECK: for (auto & MAXs_it : MAXs)
  // CHECK-NEXT: printf("s has value %d\n", (MAXs_it).x);
  // CHECK-NEXT: printf("Max of 3 and 5: %d\n", MAX(3,5));

  T DEFs;
  for (T::iterator it = DEFs.begin(), e = DEFs.end(); it != e; ++it) {
    if (*it == DEF) {
      printf("I found %d\n", *it);
    }
  }
  // CHECK: for (auto & DEFs_it : DEFs)
  // CHECK-NEXT: if (DEFs_it == DEF) {
  // CHECK-NEXT: printf("I found %d\n", DEFs_it);
}

void keywordConflict() {
  T ints;
  for (T::iterator it = ints.begin(), e = ints.end(); it != e; ++it) {
    *it = 5;
  }
  // CHECK: for (auto & ints_it : ints)
  // CHECK-NEXT: ints_it = 5;

  U __FUNCTION__s;
  for (U::iterator it = __FUNCTION__s.begin(), e = __FUNCTION__s.end();
       it != e; ++it) {
    int __FUNCTION__s_it = (*it).x + 2;
  }
  // CHECK: for (auto & __FUNCTION__s_elem : __FUNCTION__s)
  // CHECK-NEXT: int __FUNCTION__s_it = (__FUNCTION__s_elem).x + 2;
}

void typeConflict() {
  T Vals;
  // Using the name "Val", although it is the name of an existing struct, is
  // safe in this loop since it will only exist within this scope.
  for (T::iterator it = Vals.begin(), e = Vals.end(); it != e; ++it) {
  }
  // CHECK: for (auto & Val : Vals)

  // We cannot use the name "Val" in this loop since there is a reference to
  // it in the body of the loop.
  for (T::iterator it = Vals.begin(), e = Vals.end(); it != e; ++it) {
    *it = sizeof(Val);
  }
  // CHECK: for (auto & Vals_it : Vals)
  // CHECK-NEXT: Vals_it = sizeof(Val);

  typedef struct Val TD;
  U TDs;
  // Naming the variable "TD" within this loop is safe because the typedef
  // was never used within the loop.
  for (U::iterator it = TDs.begin(), e = TDs.end(); it != e; ++it) {
  }
  // CHECK: for (auto & TD : TDs)

  // "TD" cannot be used in this loop since the typedef is being used.
  for (U::iterator it = TDs.begin(), e = TDs.end(); it != e; ++it) {
    TD V;
    V.x = 5;
  }
  // CHECK: for (auto & TDs_it : TDs)
  // CHECK-NEXT: TD V;
  // CHECK-NEXT: V.x = 5;

  using ns::st;
  T sts;
  for (T::iterator it = sts.begin(), e = sts.end(); it != e; ++it) {
    *it = sizeof(st);
  }
  // CHECK: for (auto & sts_it : sts)
  // CHECK-NEXT: sts_it = sizeof(st);
}
