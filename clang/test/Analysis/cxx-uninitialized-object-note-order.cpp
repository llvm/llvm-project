// DEFINE: %{run} = %clang_analyze_cc1 \
// DEFINE:   -analyzer-checker=core,optin.cplusplus.UninitializedObject \
// DEFINE:   -analyzer-output=text -fno-caret-diagnostics %s

// RUN: %{run} -verify
// RUN: %{run} 2>&1 | FileCheck %s

// ATTENTION:
// We use FileCheck to ensure that the relative order of the notes is stable.
// These notes used to be emitted in a non-deterministic order, which is not checked by `-verify`.

struct MultipleSiblings {
  int a; // expected-note {{uninitialized field 'this->a'}}
  int b; // expected-note {{uninitialized field 'this->b'}}
  int c; // expected-note {{uninitialized field 'this->c'}}
  int d;
  MultipleSiblings() { d = 0; }
  // expected-warning@-1 {{3 uninitialized fields}}
  // expected-note@-2    {{3 uninitialized fields}}
};

void fMultipleSiblings() {
  MultipleSiblings s; // expected-note {{Calling default constructor for 'MultipleSiblings'}}
}

// CHECK-LABEL: warning: 3 uninitialized fields at the end of the constructor call
// CHECK-NEXT: note: uninitialized field 'this->a'
// CHECK-NEXT: note: uninitialized field 'this->b'
// CHECK-NEXT: note: uninitialized field 'this->c'

struct Inner {
  int x;
  // expected-note@-1 {{uninitialized field 'this->i.x'}}
  // expected-note@-2 {{uninitialized field 'this->first.x'}}
  // expected-note@-3 {{uninitialized field 'this->second.x'}}
  int y;
  // expected-note@-1 {{uninitialized field 'this->i.y'}}
  // expected-note@-2 {{uninitialized field 'this->first.y'}}
  // expected-note@-3 {{uninitialized field 'this->second.y'}}
};

struct Nested {
  Inner i;
  int z;
  Nested() { z = 0; }
  // expected-warning@-1 {{2 uninitialized fields}}
  // expected-note@-2    {{2 uninitialized fields}}
};

void fNested() {
  Nested n; // expected-note {{Calling default constructor for 'Nested'}}
}

// CHECK-LABEL: warning: 2 uninitialized fields at the end of the constructor call
// CHECK-NEXT: note: uninitialized field 'this->i.x'
// CHECK-NEXT: note: uninitialized field 'this->i.y'

struct TwoInstances {
  Inner first;
  Inner second;
  int z;
  TwoInstances() { z = 0; }
  // expected-warning@-1 {{4 uninitialized fields}}
  // expected-note@-2    {{4 uninitialized fields}}
};

void fTwoInstances() {
  TwoInstances t; // expected-note {{Calling default constructor for 'TwoInstances'}}
}

// 'first' and 'second' have the same type, so all four notes point at the two members of Inner.
// Ordering by source location alone does not separate them.
// Because of this, we sort the notes by the message as well as a tie breaker.

// CHECK-LABEL: warning: 4 uninitialized fields at the end of the constructor call
// CHECK-NEXT: note: uninitialized field 'this->first.x'
// CHECK-NEXT: note: uninitialized field 'this->second.x'
// CHECK-NEXT: note: uninitialized field 'this->first.y'
// CHECK-NEXT: note: uninitialized field 'this->second.y'

