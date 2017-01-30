// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCallGraph %s -fblocks 2>&1 | FileCheck %s

int get5() {
  return 5;
}

int add(int val1, int val2) {
  return val1 + val2;
}

int test_add() {
  return add(10, get5());
}

static void mmm(int y) {
  if (y != 0)
      y++;
  y = y/y;
}

static int foo(int x, int y) {
    mmm(y);
    if (x != 0)
      x++;
    return 5/x;
}

void aaa() {
  foo(1,2);
}

void bbb(int y) {
  int x = (y > 2);
  ^ {
      foo(x, y);
  }();
}
void ccc();
void ddd() { ccc(); }
void ccc() {}

void eee();
void eee() {}
void fff() { eee(); }

// CHECK:--- Call graph Dump ---
// CHECK-NEXT: {{Function: < root > calls: get5 add test_add mmm foo aaa < > bbb ccc ddd eee fff $}}
// CHECK-NEXT: {{Function: fff calls: eee $}}
// CHECK-NEXT: {{Function: eee calls: $}}
// CHECK-NEXT: {{Function: ddd calls: ccc $}}
// CHECK-NEXT: {{Function: ccc calls: $}}
// CHECK-NEXT: {{Function: bbb calls: < > $}}
// CHECK-NEXT: {{Function: < > calls: foo $}}
// CHECK-NEXT: {{Function: aaa calls: foo $}}
// CHECK-NEXT: {{Function: foo calls: mmm $}}
// CHECK-NEXT: {{Function: mmm calls: $}}
// CHECK-NEXT: {{Function: test_add calls: add get5 $}}
// CHECK-NEXT: {{Function: add calls: $}}
// CHECK-NEXT: {{Function: get5 calls: $}}
