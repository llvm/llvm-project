// RUN: %clang_analyze_cc1 -analyzer-checker=debug.DumpCallGraph %s 2>&1 | FileCheck %s

static int aaa() {
  return 0;
}

static int bbb(int param=aaa()) {
  return 1;
}

int ddd();

struct c {
  c(int param=2) : val(bbb(param)) {}
  int val;
  int val2 = ddd();
};

int ddd() {
  c c;
  return bbb();
}

// CHECK:--- Call graph Dump ---
// CHECK-NEXT: {{Function: < root > calls: aaa bbb c::c ddd}}
// CHECK-NEXT: {{Function: c::c calls: bbb ddd $}}
// CHECK-NEXT: {{Function: ddd calls: c::c bbb aaa $}}
// CHECK-NEXT: {{Function: bbb calls: $}}
// CHECK-NEXT: {{Function: aaa calls: $}}
