// RUN: %check_analyzer_fixit %s %t \
// RUN:   -analyzer-checker=core,optin.cplusplus.VirtualCall \
// RUN:   -analyzer-config optin.cplusplus.VirtualCall:ShowFixIts=true

struct S {
  virtual void foo();
  S() {
    foo();
    // expected-warning@-1 {{Call to virtual method 'S::foo' during construction bypasses virtual dispatch}}
    // CHECK-FIXES: S::foo();
  }
  ~S();
};
