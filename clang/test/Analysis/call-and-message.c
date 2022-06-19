// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true \
// RUN:   -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck %s

// RUN: %clang_analyze_cc1 %s -verify=no-pointee \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false

// no-pointee-no-diagnostics

void doStuff_pointerToConstInt(const int *u){};
void pointee_uninit(void) {
  int i;
  int *p = &i;
  doStuff_pointerToConstInt(p); // expected-warning{{1st function call argument is a pointer to uninitialized value [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn
// core.CallAndMessage:ArgPointeeInitializedness from a checker option into a
// checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>97a74322d64dca40aa57303842c745a1</string>
