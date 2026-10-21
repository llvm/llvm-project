// Printing parsed ClangIR input with -emit-cir reproduces the CIR it was
// emitted as.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %t.cir -o %t2.cir
// RUN: diff %t.cir %t2.cir

// Without -o, the output is named after the input with a .cir extension.
// RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir
// RUN: cp %t.cir input.txt
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir -x cir input.txt
// RUN: diff %t.cir input.cir

struct Point {
  int x;
  int y;
};

struct Point origin = {1, 2};
int counter;

static int square(int v) { return v * v; }

int add(int a, int b) { return a + b; }

int sumSquares(struct Point *p) {
  counter++;
  return add(square(p->x), square(p->y));
}
