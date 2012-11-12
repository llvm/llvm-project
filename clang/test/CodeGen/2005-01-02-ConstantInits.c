// RUN: %clang_cc1 %s -emit-llvm -o -

// This tests all kinds of hard cases with initializers and
// array subscripts.  This corresponds to PR487.

struct X { int a[2]; };

int test() {
  static int i23 = (int) &(((struct X *)0)->a[1]);
  return i23;
}

int i = (int) &( ((struct X *)0) -> a[1]);

int Arr[100];

int foo(int i) { return bar(&Arr[49])+bar(&Arr[i]); }
int foo2(int i) { 
  static const int *X = &Arr[49];
   static int i23 = (int) &( ((struct X *)0) -> a[0]);
  int *P = Arr;
  ++P;
  return bar(Arr+i);
}
