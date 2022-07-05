// RUN: %clang_extdef_map %s -- | FileCheck --implicit-check-not "c:@y" --implicit-check-not "c:@z" %s
// RUN: %clang -emit-ast %s -o %t.ast
// RUN: %clang_extdef_map %t.ast -- | FileCheck --implicit-check-not "c:@y" --implicit-check-not "c:@z" %s

int f(int) {
  return 0;
}
// CHECK-DAG: 9:c:@F@f#I#

extern const int x = 5;
// CHECK-DAG: 4:c:@x

// Non-const variables should not be collected.
int y = 5;

// In C++, const implies internal linkage, so not collected.
const int z = 5;

struct S {
  int a;
};
extern S const s = {.a = 2};
// CHECK-DAG: 4:c:@s

struct SF {
  const int a;
};
extern const SF sf = {.a = 2};
// CHECK-DAG: 5:c:@sf

struct SStatic {
  static const int a = 4;
};
const int SStatic::a;
// CHECK-DAG: 14:c:@S@SStatic@a

extern int const arr[5] = { 0, 1 };
// CHECK-DAG: 6:c:@arr

union U {
  const int a;
  const unsigned int b;
};
extern const U u = {.a = 6};
// CHECK-DAG: 4:c:@u

// No USR can be generated for this.
// Check for no crash in this case.
static union {
  float uf;
  const int ui;
};

void f(int (*)(char));
void f(bool (*)(char));

struct G {
  G() {
    f([](char) -> int { return 42; });
    // CHECK-DAG: 41:c:@S@G@F@G#@Sa@F@operator int (*)(char)#1
    f([](char) -> bool { return true; });
    // CHECK-DAG: 42:c:@S@G@F@G#@Sa@F@operator bool (*)(char)#1
  }
};
