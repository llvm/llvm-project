// RUN: %clang_cc1 -Wno-unused-value -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s
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

// <rdar://problem/14074868>
// Make sure this doesn't crash.
int foo5(bool b) {
  int y = 0;
  y = ({ A a(1); if (b) goto G; a.i; });
  G: return y;
}

// When we emit a full expression with cleanups that contains branches out of
// the full expression, the result of the inner expression (the call to
// call_with_cleanups in this case) may not dominate the fallthrough destination
// of the shared cleanup block.
//
// In this case the CFG will be a sequence of two diamonds, but the only
// dynamically possible execution paths are both left hand branches and both
// right hand branches. The first diamond LHS will call bar, and the second
// diamond LHS will assign the result to v, but the call to bar does not
// dominate the assignment.
int bar(A, int);
extern "C" int cleanup_exit_scalar(bool b) {
  int v = bar(A(1), ({ if (b) return 42; 13; }));
  return v;
}

// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_scalar({{.*}})
// CHECK: call {{.*}} @_ZN1AC1Ei
//    Spill after bar.
// CHECK: %[[v:[^ ]*]] = call{{.*}} i32 @_Z3bar1Ai({{.*}})
// CHECK-NEXT: store i32 %[[v]], i32* %[[tmp:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v:[^ ]*]] = load i32, i32* %[[tmp]]
// CHECK-NEXT: store i32 %[[v]], i32* %v

// No need to spill when the expression result is a constant, constants don't
// have dominance problems.
extern "C" int cleanup_exit_scalar_constant(bool b) {
  int v = (A(1), (void)({ if (b) return 42; 0; }), 13);
  return v;
}

// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_scalar_constant({{.*}})
// CHECK: store i32 13, i32* %v

// Check for the same bug for lvalue expression evaluation kind.
// FIXME: What about non-reference lvalues, like bitfield lvalues and vector
// lvalues?
int &getref();
extern "C" int cleanup_exit_lvalue(bool cond) {
  int &r = (A(1), ({ if (cond) return 0; (void)0; }), getref());
  return r;
}
// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_lvalue({{.*}})
// CHECK: call {{.*}} @_ZN1AC1Ei
//    Spill after bar.
// CHECK: %[[v:[^ ]*]] = call dereferenceable(4) i32* @_Z6getrefv({{.*}})
// CHECK-NEXT: store i32* %[[v]], i32** %[[tmp:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v:[^ ]*]] = load i32*, i32** %[[tmp]]
// CHECK-NEXT: store i32* %[[v]], i32** %r


// We handle ExprWithCleanups for complex evaluation type separately, and it had
// the same bug.
_Complex float bar_complex(A, int);
extern "C" int cleanup_exit_complex(bool b) {
  _Complex float v = bar_complex(A(1), ({ if (b) return 42; 13; }));
  return v;
}

// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_complex({{.*}})
// CHECK: call {{.*}} @_ZN1AC1Ei
//    Spill after bar.
// CHECK: call {{.*}} @_Z11bar_complex1Ai({{.*}})
// CHECK: store float %{{.*}}, float* %[[tmp1:[^, ]*]]
// CHECK: store float %{{.*}}, float* %[[tmp2:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v1:[^ ]*]] = load float, float* %[[tmp1]]
// CHECK: %[[v2:[^ ]*]] = load float, float* %[[tmp2]]
// CHECK: store float %[[v1]], float* %v.realp
// CHECK: store float %[[v2]], float* %v.imagp
