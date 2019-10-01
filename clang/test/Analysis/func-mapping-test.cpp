// RUN: %clang_extdef_map %s -- | FileCheck --implicit-check-not "c:@y" --implicit-check-not "c:@z" %s

int f(int) {
  return 0;
}
// CHECK-DAG: c:@F@f#I#

extern const int x = 5;
// CHECK-DAG: c:@x

// Non-const variables should not be collected.
int y = 5;

// In C++, const implies internal linkage, so not collected.
const int z = 5;

struct S {
  int a;
};
extern S const s = {.a = 2};
// CHECK-DAG: c:@s

struct SF {
  const int a;
};
SF sf = {.a = 2};
// CHECK-DAG: c:@sf

struct SStatic {
  static const int a = 4;
};
const int SStatic::a;
// CHECK-DAG: c:@S@SStatic@a

extern int const arr[5] = { 0, 1 };
// CHECK-DAG: c:@arr

union U {
  const int a;
  const unsigned int b;
};
U u = {.a = 6};
// CHECK-DAG: c:@u
