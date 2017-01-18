// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 -finline-functions -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK --check-prefix=SUITABLE
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 -finline-hint-functions -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK --check-prefix=HINTED
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-linux -O2 -fno-inline -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK --check-prefix=NOINLINE

// Force non-trivial implicit constructors/destructors/operators for B by having explicit ones for A
struct A {
  A() {}
  A(const A&) {}
  A& operator=(const A&) { return *this; }
  ~A() {}
};

struct B {
  A member;
  int implicitFunction(int a) { return a + a; }
  inline int explicitFunction(int a);
  int noHintFunction(int a);
  __attribute__((optnone)) int optNoneFunction(int a) { return a + a; }
  template<int N> int implicitTplFunction(int a) { return N + a; }
  template<int N> inline int explicitTplFunction(int a) { return N + a; }
  template<int N> int noHintTplFunction(int a);
  template<int N> int explicitRedeclTplFunction(int a);
};

int B::explicitFunction(int a) { return a + a; }
// CHECK: @_ZN1B14noHintFunctionEi({{.*}}) [[NOHINT_ATTR:#[0-9]+]]
int B::noHintFunction(int a) { return a + a; }

// CHECK: @_ZN1B19implicitTplFunctionILi0EEEii({{.*}}) [[NOHINT_ATTR]]
template<> int B::implicitTplFunction<0>(int a) { return a + a; }
// CHECK: @_ZN1B19explicitTplFunctionILi0EEEii({{.*}}) [[NOHINT_ATTR]]
template<> int B::explicitTplFunction<0>(int a) { return a + a; }
// CHECK: @_ZN1B17noHintTplFunctionILi0EEEii({{.*}}) [[NOHINT_ATTR]]
template<> int B::noHintTplFunction<0>(int a) { return a + a; }
template<> inline int B::implicitTplFunction<1>(int a) { return a; }
template<> inline int B::explicitTplFunction<1>(int a) { return a; }
template<> inline int B::noHintTplFunction<1>(int a) { return a; }
template<int N> int B::noHintTplFunction(int a) { return N + a; }
template<int N> inline int B::explicitRedeclTplFunction(int a) { return N + a; }

constexpr int constexprFunction(int a) { return a + a; }

void foo()
{
// CHECK: @_ZN1BC1Ev({{.*}}) unnamed_addr [[IMPLICIT_CONSTR_ATTR:#[0-9]+]]
  B b1;
// CHECK: @_ZN1BC1ERKS_({{.*}}) unnamed_addr [[IMPLICIT_CONSTR_ATTR]]
  B b2(b1);
// CHECK: @_ZN1BaSERKS_({{.*}}) [[IMPLICIT_CONSTR_ATTR]]
  b2 = b1;
// CHECK: @_ZN1B16implicitFunctionEi({{.*}}) [[IMPLICIT_ATTR:#[0-9]+]]
  b1.implicitFunction(1);
// CHECK: @_ZN1B16explicitFunctionEi({{.*}}) [[EXPLICIT_ATTR:#[0-9]+]]
  b1.explicitFunction(2);
  b1.noHintFunction(3);
// CHECK: @_ZN1B15optNoneFunctionEi({{.*}}) [[OPTNONE_ATTR:#[0-9]+]]
  b1.optNoneFunction(4);
// CHECK: @_Z17constexprFunctioni({{.*}}) [[IMPLICIT_ATTR]]
  constexprFunction(5);
  b1.implicitTplFunction<0>(6);
// CHECK: @_ZN1B19implicitTplFunctionILi1EEEii({{.*}}) [[EXPLICIT_ATTR]]
  b1.implicitTplFunction<1>(7);
// CHECK: @_ZN1B19implicitTplFunctionILi2EEEii({{.*}}) [[IMPLICIT_ATTR]]
  b1.implicitTplFunction<2>(8);
  b1.explicitTplFunction<0>(9);
// CHECK: @_ZN1B19explicitTplFunctionILi1EEEii({{.*}}) [[EXPLICIT_ATTR]]
  b1.explicitTplFunction<1>(10);
// CHECK: @_ZN1B19explicitTplFunctionILi2EEEii({{.*}}) [[EXPLICIT_ATTR]]
  b1.explicitTplFunction<2>(11);
  b1.noHintTplFunction<0>(12);
// CHECK: @_ZN1B17noHintTplFunctionILi1EEEii({{.*}}) [[EXPLICIT_ATTR]]
  b1.noHintTplFunction<1>(13);
// CHECK: @_ZN1B17noHintTplFunctionILi2EEEii({{.*}}) [[NOHINT_ATTR]]
  b1.noHintTplFunction<2>(14);
// CHECK: @_ZN1B25explicitRedeclTplFunctionILi2EEEii({{.*}}) [[EXPLICIT_ATTR]]
  b1.explicitRedeclTplFunction<2>(15);
// CHECK: @_ZN1BD2Ev({{.*}}) unnamed_addr [[IMPLICIT_CONSTR_ATTR]]
}

// SUITABLE-NOT: attributes [[NOHINT_ATTR]] = { {{.*}}noinline{{.*}} }
//   HINTED-DAG: attributes [[NOHINT_ATTR]] = { noinline{{.*}} }
// NOINLINE-DAG: attributes [[NOHINT_ATTR]] = { noinline{{.*}} }

// SUITABLE-NOT: attributes [[IMPLICIT_ATTR]] = { {{.*}}noinline{{.*}} }
//   HINTED-NOT: attributes [[IMPLICIT_ATTR]] = { {{.*}}noinline{{.*}} }
// NOINLINE-DAG: attributes [[IMPLICIT_ATTR]] = { noinline{{.*}} }

// SUITABLE-NOT: attributes [[IMPLICIT_CONSTR_ATTR]] = { {{.*}}noinline{{.*}} }
//   HINTED-NOT: attributes [[IMPLICIT_ATTR]] = { {{.*}}noinline{{.*}} }
// NOINLINE-DAG: attributes [[IMPLICIT_CONSTR_ATTR]] = { noinline{{.*}} }

// SUITABLE-NOT: attributes [[EXPLICIT_ATTR]] = { {{.*}}noinline{{.*}} }
//   HINTED-NOT: attributes [[IMPLICIT_ATTR]] = { {{.*}}noinline{{.*}} }
// NOINLINE-DAG: attributes [[EXPLICIT_ATTR]] = { noinline{{.*}} }

// CHECK-DAG: attributes [[OPTNONE_ATTR]] = { noinline{{.*}} }
