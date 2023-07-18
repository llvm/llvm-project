// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp 
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"__declspec(X)=" %t-rw.cpp
// rdar://11231426

// rdar://11375908
typedef unsigned long size_t;

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

void y(BOOL (^foo)());

void x() {
    y(^{
        return __objc_yes;
    });
}
