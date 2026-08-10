// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int test(int X) {
  return X;
}

void abc(int *X);
int def(int Y, int Z) {
  abc(&Z);
  return Y;
}

struct Test { short X, x; int Y, Z; };

int Testing(struct Test *A) {
  return A->X+A->Y;
}

int Test2(int X, struct Test A, int Y) {
  return X+Y+A.X+A.Y;
}
int Test3(struct Test A, struct Test B) {
  return A.X+A.Y+B.Y+B.Z;
}

struct Test Test4(struct Test A) {
  return A;
}

int Test6(void) {
  int B[200];
  return B[4];
}

struct STest2 { int X; short Y[4]; double Z; };

struct STest2 Test7(struct STest2 X) {
  return X;
}
