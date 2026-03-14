// RUN: %clang_cc1 -w -emit-llvm %s  -o /dev/null


typedef struct BF {
  int A : 1;
  char B;
  int C : 13;
} BF;

char *test1(BF *b) {
  return &b->B;        // Must be able to address non-bitfield
}

void test2(BF *b) {    // Increment and decrement operators
  b->A++;
  --b->C;
}

void test3(BF *b) {
   b->C = 12345;        // Store
}

int test4(BF *b) {
  return b->C;         // Load
}

void test5(BF *b, int i) { // array ref
  b[i].C = 12345;
}

