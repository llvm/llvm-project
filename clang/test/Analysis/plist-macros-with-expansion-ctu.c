// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -emit-pch -o %t/ctudir/plist-macros-ctu.c.ast %S/Inputs/plist-macros-ctu.c
// RUN: cp %S/Inputs/plist-macros-with-expansion-ctu.c.externalDefMap.txt %t/ctudir/externalDefMap.txt

// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config expand-macros=true \
// RUN:   -analyzer-output=plist-multi-file -o %t.plist -verify %s
// XFAIL: *
// Check the macro expansions from the plist output here, to make the test more
// understandable.
//   RUN: FileCheck --input-file=%t.plist %s

extern void F1(int **);
extern void F2(int **);
extern void F3(int **);
extern void F_H(int **);

void test0() {
  int *X;
  F3(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// CHECK: <key>name</key><string>M1</string>
// CHECK-NEXT: <key>expansion</key><string>*Z = (int *)0</string>


void test1() {
  int *X;
  F1(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>*X = (int *)0</string>

void test2() {
  int *X;
  F2(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>*Y = (int *)0</string>

#define M F1(&X)

void test3() {
  int *X;
  M;
  *X = 1; // expected-warning{{Dereference of null pointer}}
}
// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>F1(&amp;X)</string>
// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>*X = (int *)0</string>

#undef M
#define M F2(&X)

void test4() {
  int *X;
  M;
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>F2(&amp;X)</string>
// CHECK: <key>name</key><string>M</string>
// CHECK-NEXT: <key>expansion</key><string>*Y = (int *)0</string>

void test_h() {
  int *X;
  F_H(&X);
  *X = 1; // expected-warning{{Dereference of null pointer}}
}

// CHECK: <key>name</key><string>M_H</string>
// CHECK-NEXT: <key>expansion</key><string>*A = (int *)0</string>
