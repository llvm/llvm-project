// RUN: %clang_esan_frag -O0 %s -DPART1 -mllvm -esan-aux-field-info=0 -c -o %t-part1.o 2>&1
// RUN: %clang_esan_frag -O0 %s -DPART2 -c -o %t-part2.o 2>&1
// RUN: %clang_esan_frag -O0 %s -DMAIN -c -o %t-main.o 2>&1
// RUN: %clang_esan_frag -O0 %t-part1.o %t-part2.o %t-main.o -o %t 2>&1
// RUN: %env_esan_opts=verbosity=2 %run %t 2>&1 | FileCheck %s

// We generate two different object files from this file with different
// macros, and then link them together. We do this to test how we handle
// separate compilation with multiple compilation units.

#include <stdio.h>

extern "C" {
  void part1();
  void part2();
}

//===-- compilation unit part1 without main function ----------------------===//

#ifdef PART1
struct A {
  int x;
  int y;
};

struct B {
  float m;
  double n;
};

union U {
  float f;
  double d;
};

// Same struct in both main and part1.
struct S {
  int s1;
  int s2;
};

// Different structs with the same name in main and part1.
struct D {
  int d1;
  int d2;
  struct {
    int x;
    int y;
    int z;
  } ds[10];
};

void part1()
{
  struct A a;
  struct B b;
  union  U u;
  struct S s;
  struct D d;
  for (int i = 0; i < (1 << 11); i++)
    a.x = 0;
  a.y = 1;
  b.m = 2.0;
  for (int i = 0; i < (1 << 21); i++) {
    b.n = 3.0;
    d.ds[3].y = 0;
  }
  u.f = 0.0;
  u.d = 1.0;
  s.s1 = 0;
  d.d1 = 0;
}
#endif // PART1

//===-- compilation unit part2 without main function ----------------------===//
#ifdef PART2
// No struct in this part.
void part2()
{
  // do nothing
}
#endif // PART2

//===-- compilation unit with main function -------------------------------===//

#ifdef MAIN
class C {
public:
  struct {
    int x;
    int y;
  } cs;
  union {
    float f;
    double d;
  } cu;
  char c[10];
};

// Same struct in both main and part1.
struct S {
  int s1;
  int s2;
};

// Different structs with the same name in main and part1.
struct D {
  int d1;
  int d2;
  int d3;
};

int main(int argc, char **argv) {
  // CHECK:      in esan::initializeLibrary
  // CHECK:      in esan::initializeCacheFrag
  // CHECK-NEXT: in esan::processCompilationUnitInit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitInit: {{.*}}struct-simple.cpp with 6 class(es)/struct(s)
  // CHECK-NEXT:  Register struct.A$2$11$11: 2 fields
  // CHECK-NEXT:  Register struct.B$2$3$2:   2 fields
  // CHECK-NEXT:  Register union.U$1$3:      1 fields
  // CHECK-NEXT:  Register struct.S$2$11$11: 2 fields
  // CHECK-NEXT:  Register struct.D$3$14$11$11: 3 fields
  // CHECK-NEXT:  Register struct.anon$3$11$11$11: 3 fields
  // CHECK-NEXT: in esan::processCompilationUnitInit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitInit: {{.*}}struct-simple.cpp with 0 class(es)/struct(s)
  // CHECK-NEXT: in esan::processCompilationUnitInit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitInit: {{.*}}struct-simple.cpp with 5 class(es)/struct(s)
  // CHECK-NEXT:  Register class.C$3$14$13$13:  3 fields
  // CHECK-NEXT:  Register struct.anon$2$11$11: 2 fields
  // CHECK-NEXT:  Register union.anon$1$3:      1 fields
  // CHECK-NEXT:  Duplicated struct.S$2$11$11:  2 fields
  // CHECK-NEXT:  Register struct.D$3$11$11$11: 3 fields
  struct C c[2];
  struct S s;
  struct D d;
  c[0].cs.x = 0;
  c[1].cs.y = 1;
  c[0].cu.f = 0.0;
  c[1].cu.d = 1.0;
  c[0].c[2] = 0;
  s.s1 = 0;
  d.d1 = 0;
  d.d2 = 0;
  part1();
  part2();
  return 0;
  // CHECK:      in esan::finalizeLibrary
  // CHECK-NEXT: in esan::finalizeCacheFrag
  // CHECK-NEXT: in esan::processCompilationUnitExit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitExit: {{.*}}struct-simple.cpp with 5 class(es)/struct(s)
  // CHECK-NEXT:  Unregister class.C$3$14$13$13:  3 fields
  // CHECK-NEXT:   {{.*}} class C
  // CHECK-NEXT:   {{.*}}  size = 32, count = 5, ratio = 3, array access = 5
  // CHECK-NEXT:   {{.*}}  # 0: offset = 0,  size = 8,  count = 2, type = %struct.anon = type { i32, i32 }
  // CHECK-NEXT:   {{.*}}  # 1: offset = 8,  size = 8,  count = 2, type = %union.anon = type { double }
  // CHECK-NEXT:   {{.*}}  # 2: offset = 16, size = 10, count = 1, type = [10 x i8]
  // CHECK-NEXT:  Unregister struct.anon$2$11$11: 2 fields
  // CHECK-NEXT:   {{.*}} struct anon
  // CHECK-NEXT:   {{.*}}  size = 8, count = 2, ratio = 1, array access = 0
  // CHECK-NEXT:   {{.*}}  # 0: offset = 0, size = 4, count = 1, type = i32
  // CHECK-NEXT:   {{.*}}  # 1: offset = 4, size = 4, count = 1, type = i32
  // CHECK-NEXT:  Unregister union.anon$1$3:      1 fields
  // CHECK-NEXT:  Unregister struct.S$2$11$11:    2 fields
  // CHECK-NEXT:   {{.*}} struct S
  // CHECK-NEXT:   {{.*}}  size = 8, count = 2, ratio = 2, array access = 0
  // CHECK-NEXT:   {{.*}}  # 0: count = 2
  // CHECK-NEXT:   {{.*}}  # 1: count = 0
  // CHECK-NEXT:  Unregister struct.D$3$11$11$11: 3 fields
  // CHECK-NEXT:   {{.*}} struct D
  // CHECK-NEXT:   {{.*}}  size = 12, count = 2, ratio = 2, array access = 0
  // CHECK-NEXT:   {{.*}}  # 0: offset = 0, size = 4, count = 1, type = i32
  // CHECK-NEXT:   {{.*}}  # 1: offset = 4, size = 4, count = 1, type = i32
  // CHECK-NEXT:   {{.*}}  # 2: offset = 8, size = 4, count = 0, type = i32
  // CHECK-NEXT: in esan::processCompilationUnitExit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitExit: {{.*}}struct-simple.cpp with 0 class(es)/struct(s)
  // CHECK-NEXT: in esan::processCompilationUnitExit
  // CHECK-NEXT: in esan::processCacheFragCompilationUnitExit: {{.*}}struct-simple.cpp with 6 class(es)/struct(s)
  // CHECK-NEXT:  Unregister struct.A$2$11$11:    2 fields
  // CHECK-NEXT:   {{.*}} struct A
  // CHECK-NEXT:   {{.*}}  size = 8, count = 2049, ratio = 2048, array access = 0
  // CHECK-NEXT:   {{.*}}  # 0: count = 2048
  // CHECK-NEXT:   {{.*}}  # 1: count = 1
  // CHECK-NEXT:  Unregister struct.B$2$3$2:      2 fields
  // CHECK-NEXT:   {{.*}} struct B
  // CHECK-NEXT:   {{.*}}  size = 16, count = 2097153, ratio = 2097152, array access = 0
  // CHECK-NEXT:   {{.*}}  # 0: count = 1
  // CHECK-NEXT:   {{.*}}  # 1: count = 2097152
  // CHECK-NEXT:  Unregister union.U$1$3:         1 fields
  // CHECK-NEXT:  Duplicated struct.S$2$11$11:    2 fields
  // CHECK-NEXT:  Unregister struct.D$3$14$11$11: 3 fields
  // CHECK-NEXT:  {{.*}} struct D
  // CHECK-NEXT:  {{.*}}  size = 128, count = 2097153, ratio = 2097153, array access = 0
  // CHECK-NEXT:  {{.*}}  # 0: count = 1
  // CHECK-NEXT:  {{.*}}  # 1: count = 0
  // CHECK-NEXT:  {{.*}}  # 2: count = 2097152
  // CHECK-NEXT:  Unregister struct.anon$3$11$11$11: 3 fields
  // CHECK-NEXT:  {{.*}} struct anon
  // CHECK-NEXT:  {{.*}}  size = 12, count = 2097152, ratio = 4194304, array access = 2097152
  // CHECK-NEXT:  {{.*}}  # 0: count = 0
  // CHECK-NEXT:  {{.*}}  # 1: count = 2097152
  // CHECK-NEXT:  {{.*}}  # 2: count = 0
  // CHECK-NEXT: {{.*}}EfficiencySanitizer: total struct field access count = 6293518
}
#endif // MAIN
