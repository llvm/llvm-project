// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


#define A(X) int X;
#define B(X) A(X##0) A(X##1) A(X##2) A(X##3) A(X##4) A(X##5) A(X##6) A(X##7) \
             A(X##8) A(X##9) A(X##A) A(X##B) A(X##C) A(X##D) A(X##E) A(X##F)
#define C(X) B(X##0) B(X##1) B(X##2) B(X##3) B(X##4) B(X##5) B(X##6) B(X##7) \
             B(X##8) B(X##9) B(X##A) B(X##B) B(X##C) B(X##D) B(X##E) B(X##F)

struct foo {
  C(x);   // 256
  C(y);   // 256
  C(z);
};


int test(struct foo *F) {
   return F->xA1 + F->yFF + F->zC4;
}
