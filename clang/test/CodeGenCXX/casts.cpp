// RUN: %clang_cc1 %s -emit-llvm -o %t

// PR5248
namespace PR5248 {
struct A {
  void copyFrom(const A &src);
  void addRef(void);

  A& operator=(int);
};

void A::copyFrom(const A &src) {
  ((A &)src).addRef();
}
}

// reinterpret_cast to self
void test(PR5248::A* a) {
  reinterpret_cast<PR5248::A&>(*a) = 17;
}
