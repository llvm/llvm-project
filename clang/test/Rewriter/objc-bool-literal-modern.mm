// RUN: %clang_cc1 -x objective-c++ -fms-extensions -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only -D"__declspec(X)=" %t-rw.cpp
// rdar://11124775

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
