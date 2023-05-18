// RUN: %clang_cc1 %s -emit-llvm -o %t

extern "C" int printf(...);

static int val;

struct B {
  B() : iB(++val) { printf("B()\n"); }
  int iB;
  ~B() { printf("~B(%d)\n", iB); --val; }
};

struct M : B {
  M() : iM(++val) { printf("M()\n"); }
  int iM;
  ~M() { printf("~M(%d)\n", iM); --val; }
};

struct P {
  P() : iP(++val) { printf("P()\n"); }
  int iP;
  ~P() { printf("~P(%d)\n", iP); --val; }
};

struct N : M, P {
  N() { printf("N()\n"); iN = ++val; }
  ~N() { printf("~N(%d) val = %d\n", iN, --val);  }
  int iN;
  M m;
  P p;
};

struct O : B { 
  ~O() { return; } 
};

int main() {
  N n1;
  N n2;
  O o;
}
