// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -emit-pch -o %t/ctudir/plist-macros-ctu.c.ast %S/Inputs/plist-macros-ctu.c
// RUN: cp %S/Inputs/plist-macros-with-expansion-ctu.c.externalDefMap.txt %t/ctudir/externalDefMap.txt
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config expand-macros=true \
// RUN:   -analyzer-output=plist-multi-file -o %t.plist -verify %s
//
// Check the macro expansions from the plist output here, to make the test more
// understandable.
//   RUN: FileCheck --input-file=%t.plist %s

extern void F1(int **);
extern void F2(int **);
extern void F3(int **);
extern void F_H(int **);

void test0(void) {
  int *X;
  F3(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// FIXME: Macro expansion for other TUs should also work.
// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>

void test1(void) {
  int *X;
  F1(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>

void test2(void) {
  int *X;
  F2(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>

#define M F1(&X)

void test3(void) {
  int *X;
  M;
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// Macro expansions for the main TU still works, even in CTU mode.
// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>55</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>M</string>
// CHECK-NEXT:   <key>expansion</key><string>F1 (&amp;X )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

#undef M
#define M F2(&X)

void test4(void) {
  int *X;
  M;
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT:  <dict>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>78</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <key>name</key><string>M</string>
// CHECK-NEXT:   <key>expansion</key><string>F2 (&amp;X )</string>
// CHECK-NEXT:  </dict>
// CHECK-NEXT: </array>

void test_h(void) {
  int *X;
  F_H(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK:      <key>macro_expansions</key>
// CHECK-NEXT: <array>
// CHECK-NEXT: </array>
