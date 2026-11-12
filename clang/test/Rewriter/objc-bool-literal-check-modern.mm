// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 

typedef bool BOOL;

BOOL yes() {
  return __objc_yes;
}

BOOL no() {
  return __objc_no;
}

BOOL which (int flag) {
  return flag ? yes() : no();
}

int main() {
  which (__objc_yes);
  which (__objc_no);
  return __objc_yes;
}

// CHECK: return ((bool)1);
// CHECK: return ((bool)0);
// CHECK: which (((bool)1));
// CHECK: which (((bool)0));
// CHECK: return ((bool)1);
