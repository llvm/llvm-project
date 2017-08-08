// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -fblocks -verify %s

int* f1() {
  int x = 0;
  return &x; // expected-warning{{Address of stack memory associated with local variable 'x' returned}} expected-warning{{address of stack memory associated with local variable 'x' returned}}
}

int* f2(int y) {
  return &y;  // expected-warning{{Address of stack memory associated with local variable 'y' returned}} expected-warning{{address of stack memory associated with local variable 'y' returned}}
}

int* f3(int x, int *y) {
  int w = 0;
  
  if (x)
    y = &w;
    
  return y; // expected-warning{{Address of stack memory associated with local variable 'w' returned to caller}}
}

void* compound_literal(int x, int y) {
  if (x)
    return &(unsigned short){((unsigned short)0x22EF)}; // expected-warning{{Address of stack memory}}

  int* array[] = {};
  struct s { int z; double y; int w; };
  
  if (y)
    return &((struct s){ 2, 0.4, 5 * 8 }); // expected-warning{{Address of stack memory}}
    
  
  void* p = &((struct s){ 42, 0.4, x ? 42 : 0 });
  return p; // expected-warning{{Address of stack memory}}
}

void* alloca_test() {
  void* p = __builtin_alloca(10);
  return p; // expected-warning{{Address of stack memory}}
}

int array_test(int x[2]) {
  return x[0]; // no-warning
}

struct baz {
  int x;
  int y[2];
};

int struct_test(struct baz byVal, int flag) {
  if (flag)  
    return byVal.x; // no-warning
  else {
    return byVal.y[0]; // no-warning
  }
}

typedef int (^ComparatorBlock)(int a, int b);
ComparatorBlock test_return_block(void) {
  // This block is a global since it has no captures.
  ComparatorBlock b = ^int(int a, int b){ return a > b; };
  return b; // no-warning
}

ComparatorBlock test_return_block_with_capture(int x) {
  // This block is stack allocated because it has captures.
  ComparatorBlock b = ^int(int a, int b){ return a > b + x; };
  return b; // expected-warning{{Address of stack-allocated block}}
}

ComparatorBlock test_return_block_neg_aux(void);
ComparatorBlock test_return_block_neg(void) {
  ComparatorBlock b = test_return_block_neg_aux();
  return b; // no-warning
}

// <rdar://problem/7523821>
int *rdar_7523821_f2() {
  int a[3];
  return a; // expected-warning 2 {{ddress of stack memory associated with local variable 'a' returned}}
};

// Handle blocks that have no captures or are otherwise declared 'static'.
// <rdar://problem/10348049>
typedef int (^RDar10348049)(int value);
RDar10348049 test_rdar10348049(void) {
  static RDar10348049 b = ^int(int x) {
    return x + 2;
  };
  return b; // no-warning
}

void testRegister(register const char *reg) {
    if (reg) (void)reg[0];
}
void callTestRegister() {
    char buf[20];
    testRegister(buf); // no-warning
}
