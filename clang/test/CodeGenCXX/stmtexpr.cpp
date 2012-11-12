// RUN: %clang_cc1 -Wno-unused-value -emit-llvm -o - %s | FileCheck %s
// rdar: //8540501
extern "C" int printf(...);
extern "C" void abort();

struct A
{
  int i;
  A (int j) : i(j) {printf("this = %p A(%d)\n", this, j);}
  A (const A &j) : i(j.i) {printf("this = %p const A&(%d)\n", this, i);}
  A& operator= (const A &j) { i = j.i; abort(); return *this; }
  ~A() { printf("this = %p ~A(%d)\n", this, i); }
};

struct B
{
  int i;
  B (const A& a) { i = a.i; }
  B() {printf("this = %p B()\n", this);}
  B (const B &j) : i(j.i) {printf("this = %p const B&(%d)\n", this, i);}
  ~B() { printf("this = %p ~B(%d)\n", this, i); }
};

A foo(int j)
{
  return ({ j ? A(1) : A(0); });
}


void foo2()
{
  A b = ({ A a(1); A a1(2); A a2(3); a1; a2; a; });
  if (b.i != 1)
    abort(); 
  A c = ({ A a(1); A a1(2); A a2(3); a1; a2; a; A a3(4); a2; a3; });
  if (c.i != 4)
    abort(); 
}

void foo3()
{
  const A &b = ({ A a(1); a; });
  if (b.i != 1)
    abort();
}

void foo4()
{
// CHECK: call {{.*}} @_ZN1AC1Ei
// CHECK: call {{.*}} @_ZN1AC1ERKS_
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: call {{.*}} @_ZN1BC1ERK1A
// CHECK: call {{.*}} @_ZN1AD1Ev
  const B &b = ({ A a(1); a; });
  if (b.i != 1)
    abort();
}

int main()
{
  foo2();
  foo3();
  foo4();
  return foo(1).i-1;
}

// rdar: // 8600553
int a[128];
int* foo5() {
// CHECK-NOT: memcpy
  // Check that array-to-pointer conversion occurs in a
  // statement-expression.
  return (({ a; }));
}

