// Test to ensure RHS condition of logical operators isn't evaluated more than
// one time when instrumenting RHS counter blocks for branch coverage.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -main-file-name branch-profdup.cpp %s -o - -emit-llvm -fprofile-instrument=clang | FileCheck -allow-deprecated-dag-overlap %s

// CHECK-LABEL: define {{.*}}@_Z5test1b
// CHECK-COUNT-1: = call {{.*}}@_Z5fval1v()
// CHECK-NOT: = call {{.*}}@_Z5fval1v()
extern bool fval1();
bool test1(bool a) {
  return (a && fval1());
}

// CHECK-LABEL: define {{.*}}@_Z5test2b
// CHECK-COUNT-1: call {{.*}}_Z5fval2v()
// CHECK-NOT: call {{.*}}_Z5fval2v()
extern bool fval2();
bool test2(bool a) {
  return (a || fval2());
}

// CHECK-LABEL: define {{.*}}@_Z5test3v
// CHECK-COUNT-1: call {{.*}}_Z5fval3v()
// CHECK-NOT: call {{.*}}_Z5fval3v()
extern bool fval3();
bool test3() {
  return (1 && fval3());
}

// CHECK-LABEL: define {{.*}}@_Z5test4v
// CHECK-COUNT-1: call {{.*}}_Z5fval4v()
// CHECK-NOT: call {{.*}}_Z5fval4v()
extern bool fval4();
bool test4() {
  return (0 || fval4());
}

// CHECK-LABEL: define {{.*}}@_Z5test5b
// CHECK-COUNT-1: call {{.*}}_Z5fval5v()
// CHECK-NOT: call {{.*}}_Z5fval5v()
extern bool fval5();
bool test5(bool a) {
  if (a && fval5())
    return true;
  return false;
}

// CHECK-LABEL: define {{.*}}@_Z5test6b
// CHECK-COUNT-1: call {{.*}}_Z5fval6v()
// CHECK-NOT: call {{.*}}_Z5fval6v()
extern bool fval6();
bool test6(bool a) {
  if (a || fval6())
    return true;
  return false;
}

// CHECK-LABEL: define {{.*}}@_Z5test7v
// CHECK-COUNT-1: call {{.*}}_Z5fval7v()
// CHECK-NOT: call {{.*}}_Z5fval7v()
extern bool fval7();
bool test7() {
  if (1 && fval7())
    return true;
  return false;
}

// CHECK-LABEL: define {{.*}}@_Z5test8v
// CHECK-COUNT-1: call {{.*}}_Z5fval8v()
// CHECK-NOT: call {{.*}}_Z5fval8v()
extern bool fval8();
bool test8() {
  if (0 || fval8())
    return true;
  return false;
}
