// RUN: %clang_cc1 -emit-llvm-only %s
void f(bool flag) {
  int a = 1;
  int b = 2;
  
  (flag ? a : b) = 3;
}

// PR10756
namespace test0 {
  struct A {
    A(const A &);
    A &operator=(const A &);
    A sub() const;
    void foo() const;
  };
  void foo(bool cond, const A &a) {
    (cond ? a : a.sub()).foo();
  }
}
