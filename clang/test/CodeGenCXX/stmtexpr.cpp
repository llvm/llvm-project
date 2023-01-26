// RUN: %clang_cc1 -Wno-unused-value -triple i686-linux-gnu -emit-llvm -o - %s | FileCheck %s
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
// CHECK-NEXT: store i32 %[[v]], ptr %[[tmp:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v:[^ ]*]] = load i32, ptr %[[tmp]]
// CHECK-NEXT: store i32 %[[v]], ptr %v

// No need to spill when the expression result is a constant, constants don't
// have dominance problems.
extern "C" int cleanup_exit_scalar_constant(bool b) {
  int v = (A(1), (void)({ if (b) return 42; 0; }), 13);
  return v;
}

// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_scalar_constant({{.*}})
// CHECK: store i32 13, ptr %v

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
// CHECK: %[[v:[^ ]*]] = call noundef nonnull align 4 dereferenceable(4) ptr @_Z6getrefv({{.*}})
// CHECK-NEXT: store ptr %[[v]], ptr %[[tmp:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v:[^ ]*]] = load ptr, ptr %[[tmp]]
// CHECK-NEXT: store ptr %[[v]], ptr %r

// Bind the reference to a byval argument. It is not an instruction or Constant,
// so it's a bit of a corner case.
struct ByVal { int x[3]; };
extern "C" int cleanup_exit_lvalue_byval(bool cond, ByVal arg) {
  ByVal &r = (A(1), ({ if (cond) return 0; (void)ByVal(); }), arg);
  return r.x[0];
}
// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_lvalue_byval({{.*}}, ptr noundef byval(%struct.ByVal) align 4 %arg)
// CHECK: call {{.*}} @_ZN1AC1Ei
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
// CHECK: store ptr %arg, ptr %r

// Bind the reference to a local variable. We don't need to spill it. Binding a
// reference to it doesn't generate any instructions.
extern "C" int cleanup_exit_lvalue_local(bool cond) {
  int local = 42;
  int &r = (A(1), ({ if (cond) return 0; (void)0; }), local);
  return r;
}
// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_lvalue_local({{.*}})
// CHECK: %local = alloca i32
// CHECK: store i32 42, ptr %local
// CHECK: call {{.*}} @_ZN1AC1Ei
// CHECK-NOT: store ptr %local
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
// CHECK: store ptr %local, ptr %r, align 4

// We handle ExprWithCleanups for complex evaluation type separately, and it had
// the same bug.
_Complex float bar_complex(A, int);
extern "C" int cleanup_exit_complex(bool b) {
  _Complex float v = bar_complex(A(1), ({ if (b) return 42; 13; }));
  return (float)v;
}

// CHECK-LABEL: define{{.*}} i32 @cleanup_exit_complex({{.*}})
// CHECK: call {{.*}} @_ZN1AC1Ei
//    Spill after bar.
// CHECK: call {{.*}} @_Z11bar_complex1Ai({{.*}})
// CHECK: store float %{{.*}}, ptr %[[tmp1:[^, ]*]]
// CHECK: store float %{{.*}}, ptr %[[tmp2:[^, ]*]]
//    Do cleanup.
// CHECK: call {{.*}} @_ZN1AD1Ev
// CHECK: switch
//    Reload before v assignment.
// CHECK: %[[v1:[^ ]*]] = load float, ptr %[[tmp1]]
// CHECK: %[[v2:[^ ]*]] = load float, ptr %[[tmp2]]
// CHECK: store float %[[v1]], ptr %v.realp
// CHECK: store float %[[v2]], ptr %v.imagp

extern "C" void then(int);

// CHECK-LABEL: @{{.*}}volatile_load
void volatile_load() {
  volatile int n;

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({n;});

  // CHECK-LABEL: @then(i32 noundef 1)
  then(1);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({goto lab; lab: n;});

  // CHECK-LABEL: @then(i32 noundef 2)
  then(2);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({[[gsl::suppress("foo")]] n;});

  // CHECK-LABEL: @then(i32 noundef 3)
  then(3);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({if (true) n;});

  // CHECK: }
}

// CHECK-LABEL: @{{.*}}volatile_load_template
template<typename T>
void volatile_load_template() {
  volatile T n;

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({n;});

  // CHECK-LABEL: @then(i32 noundef 1)
  then(1);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({goto lab; lab: n;});

  // CHECK-LABEL: @then(i32 noundef 2)
  then(2);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({[[gsl::suppress("foo")]] n;});

  // CHECK-LABEL: @then(i32 noundef 3)
  then(3);

  // CHECK-NOT: load volatile
  // CHECK: load volatile
  // CHECK-NOT: load volatile
  ({if (true) n;});

  // CHECK: }
}
template void volatile_load_template<int>();
