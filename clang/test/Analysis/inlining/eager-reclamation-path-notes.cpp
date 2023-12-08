// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config graph-trim-interval=5 -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file -analyzer-config graph-trim-interval=5 -analyzer-config suppress-null-return-paths=false %s -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/eager-reclamation-path-notes.cpp.plist -

struct IntWrapper {
  int getValue();
};

IntWrapper *getNullWrapper() {
  return 0;
  // expected-note@-1 {{Returning null pointer}}
}

int memberCallBaseDisappears() {
  // In this case, we need the lvalue-to-rvalue cast for 'ptr' to disappear,
  // which means we need to trigger reclamation between that and the ->
  // operator.
  //
  // Note that this test is EXTREMELY brittle because it's a negative test:
  // we want to show that even if the node for the rvalue of 'ptr' disappears,
  // we get the same results as if it doesn't. The test should never fail even
  // if our node reclamation policy changes, but it could easily not be testing
  // anything at that point.
  IntWrapper *ptr = getNullWrapper();
  // expected-note@-1 {{Calling 'getNullWrapper'}}
  // expected-note@-2 {{Returning from 'getNullWrapper'}}
  // expected-note@-3 {{'ptr' initialized to a null pointer value}}

  // Burn some nodes to trigger reclamation.
  int unused = 1;
  (void)unused;

  return ptr->getValue(); // expected-warning {{Called C++ object pointer is null}}
  // expected-note@-1 {{Called C++ object pointer is null}}
}

