// RUN: %clang_analyze_cc1 -analyzer-checker=core.uninitialized.Branch \
// RUN:   -analyzer-output=text -verify %s

// Test that 'initialized to a garbage value' note is emitted when a variable
// is initialized from an uninitialized source and later used in a branch.
void testInitFromGarbage(int cond) {
  int y;             // expected-note{{'y' declared without an initial value}}
  if (cond)          // expected-note{{Assuming 'cond' is 0}}
                     // expected-note@-1{{Taking false branch}}
    y = 1;
  int x = y;         // expected-note{{'x' initialized to a garbage value}}
  if (x > 0) {}     // expected-warning{{Branch condition evaluates to a garbage value}}
                     // expected-note@-1{{Branch condition evaluates to a garbage value}}
}
