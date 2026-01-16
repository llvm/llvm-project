// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck %s

// RUN: %clang_analyze_cc1 %s -verify=no-pointee \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false

// RUN: %clang_analyze_cc1 %s -verify=arg-init \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=true

// no-pointee-no-diagnostics

void doStuff_pointerToConstInt(const int *u){};
void pointee_uninit(void) {
  int i;
  int *p = &i;
  doStuff_pointerToConstInt(p); // expected-warning{{1st function call argument is a pointer to uninitialized value [core.CallAndMessage]}}
}

typedef struct S {
  int a;
  short b;
} S;

void doStuff_pointerToConstStruct(const S *s){};
void pointee_uninit_struct(void) {
  S s;
  S *p = &s;
  doStuff_pointerToConstStruct(p); // expected-warning{{1st function call argument points to an uninitialized value (e.g., field: 'a') [core.CallAndMessage]}}
}
void pointee_uninit_struct_1(void) {
  S s;
  s.a = 2;
  doStuff_pointerToConstStruct(&s); // expected-warning{{1st function call argument points to an uninitialized value (e.g., field: 'b') [core.CallAndMessage]}}
}
void pointee_uninit_struct_2(void) {
  S s = {};
  doStuff_pointerToConstStruct(&s);
}
void pointee_uninit_struct_3(S *s) {
  doStuff_pointerToConstStruct(s);
}
void pointee_uninit_struct_4(void) {
  S s = {1, 2};
  doStuff_pointerToConstStruct(&s);
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:ArgPointeeInitializedness from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>97a74322d64dca40aa57303842c745a1</string>

typedef struct {
  int i  :2;
  int    :30;  // unnamed bit-field
} B;

extern void consume_B(B);

void bitfield_B_init(void) {
  B b1;
  b1.i = 1; // b1 is initialized
  consume_B(b1);
}

void bitfield_B_uninit(void) {
  B b2;
  consume_B(b2); // arg-init-warning{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'i') [core.CallAndMessage]}}
}
