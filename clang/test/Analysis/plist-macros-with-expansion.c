// RUN: %clang_analyze_cc1 -analyzer-checker=core %s  \
// RUN:   -analyzer-output=plist -o %t.plist \
// RUN:   -analyzer-config expand-macros=true -verify
//
// RUN: FileCheck --input-file=%t.plist %s

#define STRANGE_FN(x) STRANGE_FN(x, 0)
void test_strange_macro_expansion() {
  char *path;
  STRANGE_FN(path); // no-crash
  // expected-warning@-1 {{implicit declaration of function}}
  // expected-warning@-2 {{1st function call argument is an uninitialized value}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>10</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>STRANGE_FN(path)</string>
// CHECK-NEXT:   <key>expansion</key><string>STRANGE_FN (path ,0)</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

