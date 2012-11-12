// Test returning a single element aggregate value containing a double.
// RUN: %clang_cc1 %s -emit-llvm -o -

struct X {
  double D;
};

struct Y { 
  struct X x; 
};

struct Y bar();

void foo(struct Y *P) {
  *P = bar();
}

struct Y bar() {
  struct Y a;
  a.x.D = 0;
  return a;
}

