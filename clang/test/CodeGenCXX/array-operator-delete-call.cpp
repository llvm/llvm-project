// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

extern "C" int printf(...);

int count;

struct S {
  S() : iS (++count) { printf("S::S(%d)\n", iS); }
  ~S() { printf("S::~S(%d)\n", iS); }
  int iS;
};

struct V {
  V() : iV (++count) { printf("V::V(%d)\n", iV); }
  virtual ~V() { printf("V::~V(%d)\n", iV); }
  int iV;
};

struct COST
{
  S *cost;
  V *vcost;
  unsigned *cost_val;

  ~COST();
  COST();
};


COST::COST()
{
  cost = new S[3];
  vcost = new V[4];
  cost_val = new unsigned[10];
}

COST::~COST()
{
  if (cost) {
   delete [] cost;
  }
  if (vcost) {
   delete [] vcost;
  }
  if (cost_val)
    delete [] cost_val;
}

COST c1;

int main()
{
  COST c3;
}
COST c2;

// CHECK-LP64: callq    __ZdaPv

// CHECK-LP32: calll     L__ZdaPv

