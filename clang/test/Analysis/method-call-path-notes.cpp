// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file  %s -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/method-call-path-notes.cpp.plist -

// Test warning about null or uninitialized pointer values used as instance member
// calls.
class TestInstanceCall {
public:
  void foo() {}
};

void test_ic() {
  TestInstanceCall *p; // expected-note {{'p' declared without an initial value}}
  p->foo(); // expected-warning {{Called C++ object pointer is uninitialized}} expected-note {{Called C++ object pointer is uninitialized}}
}

void test_ic_null() {
  TestInstanceCall *p = 0; // expected-note {{'p' initialized to a null pointer value}}
  p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

void test_ic_set_to_null() {
  TestInstanceCall *p;
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
  p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

void test_ic_null(TestInstanceCall *p) {
  if (!p) // expected-note {{Assuming 'p' is null}} expected-note {{Taking true branch}}
    p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note{{Called C++ object pointer is null}}
}

void test_ic_member_ptr() {
  TestInstanceCall *p = 0; // expected-note {{'p' initialized to a null pointer value}}
  typedef void (TestInstanceCall::*IC_Ptr)();
  IC_Ptr bar = &TestInstanceCall::foo;
  (p->*bar)(); // expected-warning {{Called C++ object pointer is null}} expected-note{{Called C++ object pointer is null}}
}

void test_cast(const TestInstanceCall *p) {
  if (!p) // expected-note {{Assuming 'p' is null}} expected-note {{Taking true branch}}
    const_cast<TestInstanceCall *>(p)->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

