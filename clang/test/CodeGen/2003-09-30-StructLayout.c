// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

enum En {
  ENUM_VAL
};

struct St {
  unsigned char A;
  enum En B;
  unsigned char C;
  enum En D;
  float E;
};


void func(struct St* A) {
  A->D = ENUM_VAL;
}
