// RUN: %clang_cc1 -emit-llvm -o - %s
// pr5547

struct A {
  void* operator new(__typeof(sizeof(int)));
  A();
};

A* x() {
  return new A;
}

struct B {
  void* operator new(__typeof(sizeof(int)), int = 1, int = 4);
  B(float);
};

B* y() {
  new (3,4) B(1);
  return new(1) B(2);
}

struct C {
  void* operator new(__typeof(sizeof(int)), int, int = 4);
  C();
};

C* z() {
  new (3,4) C;
  return new(1) C;
}


