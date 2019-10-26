// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus \
// RUN:       -analyzer-output=plist -o %t.plist -w -verify=pure %s
// RUN: cat %t.plist | FileCheck --check-prefixes=PURE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,optin.cplusplus \
// RUN:       -analyzer-output=plist -o %t.plist -w -verify=impure %s
// RUN: cat %t.plist | FileCheck --check-prefixes=IMPURE %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus,optin.cplusplus \
// RUN:       -analyzer-output=plist -o %t.plist -w -verify=pure,impure %s
// RUN: cat %t.plist | FileCheck --check-prefixes=PURE,IMPURE %s

struct S {
  virtual void foo();
  virtual void bar() = 0;

  S() {
    // IMPURE: Call to virtual method &apos;S::foo&apos; during construction bypasses virtual dispatch
    // IMPURE: optin.cplusplus.VirtualCall
    foo(); // impure-warning{{Call to virtual method 'S::foo' during construction bypasses virtual dispatch}}
    // PURE: Call to pure virtual method &apos;S::bar&apos; during construction has undefined behavior
    // PURE: cplusplus.PureVirtualCall
    bar(); // pure-warning{{Call to pure virtual method 'S::bar' during construction has undefined behavior}}
  }
};
