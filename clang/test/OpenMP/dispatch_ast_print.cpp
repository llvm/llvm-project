// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fopenmp -fopenmp-version=51     \
// RUN:   -fsyntax-only -verify %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -ast-dump  %s | FileCheck %s --check-prefix=DUMP

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

#ifndef HEADER
#define HEADER

int foo_gpu(int A, int *B) { return 0;}
//PRINT: #pragma omp declare variant(foo_gpu)
//DUMP: FunctionDecl{{.*}} foo
//DUMP: OMPDeclareVariantAttr {{.*}}Implicit construct{{.*}}
#pragma omp declare variant(foo_gpu) \
    match(construct={dispatch}, device={arch(arm)})
int foo(int, int*);

template <typename T, typename TP>
void fooTemp() {
  T a;
  TP b;
  //PRINT: #pragma omp dispatch nowait
  //DUMP: OMPDispatchDirective
  //DUMP: OMPNowaitClause
  #pragma omp dispatch nowait
  foo(a, b);
}

int *get_device_ptr();
int get_device();
int other();

//DUMP: FunctionDecl{{.*}} test_one
void test_one()
{
  int aaa, bbb, var;
  //PRINT: #pragma omp dispatch depend(in : var) nowait novariants(aaa > 5) nocontext(bbb > 5)
  //DUMP: OMPDispatchDirective
  //DUMP: OMPDependClause
  //DUMP: OMPNowaitClause
  //DUMP: OMPNovariantsClause
  #pragma omp dispatch depend(in:var) nowait novariants(aaa > 5) nocontext(bbb > 5)
  foo(aaa, &bbb);

  int *dp = get_device_ptr();
  int dev = get_device();
  //PRINT: #pragma omp dispatch device(dev) is_device_ptr(dp) novariants(dev > 10) nocontext(dev > 5)
  //DUMP: OMPDispatchDirective
  //DUMP: OMPDeviceClause
  //DUMP: OMPIs_device_ptrClause
  //DUMP: OMPNovariantsClause
  #pragma omp dispatch device(dev) is_device_ptr(dp) novariants(dev > 10) nocontext(dev > 5)
  foo(aaa, dp);

  //PRINT: #pragma omp dispatch
  //PRINT: foo(other(), &bbb);
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  foo(other(), &bbb);

  fooTemp<int, int*>();
}

struct Obj {
  Obj();
  ~Obj();
  int disp_method_variant1();
  #pragma omp declare variant(disp_method_variant1)                            \
    match(construct={dispatch}, device={arch(arm)})
  int disp_method1();

  static int disp_method_variant2() { return 1; }
  #pragma omp declare variant(disp_method_variant2)                            \
    match(construct={dispatch}, device={arch(arm)})
  static int disp_method2() { return 2; }
};

Obj foo_vari();
#pragma omp declare variant(foo_vari) \
  match(construct={dispatch}, device={arch(arm)})
Obj foo_obj();

//DUMP: FunctionDecl{{.*}} test_two
void test_two(Obj o1, Obj &o2, Obj *o3)
{
  //PRINT: #pragma omp dispatch
  //PRINT: o1.disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  o1.disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: o2.disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  o2.disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: o3->disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  o3->disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: Obj::disp_method2();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  Obj::disp_method2();

  int ret;
  //PRINT: #pragma omp dispatch
  //PRINT: ret = o1.disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  ret = o1.disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: ret = o2.disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  ret = o2.disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: ret = o3->disp_method1();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  ret = o3->disp_method1();

  //PRINT: #pragma omp dispatch
  //PRINT: ret = Obj::disp_method2();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  ret = Obj::disp_method2();

  //PRINT: #pragma omp dispatch
  //PRINT: (void)Obj::disp_method2();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  (void)Obj::disp_method2();

  // Full C++ operator= case with temps and EH.
  Obj o;
  //PRINT: #pragma omp dispatch
  //PRINT: o = foo_obj();
  //DUMP: OMPDispatchDirective
  #pragma omp dispatch
  o = foo_obj();
}

struct A {
  A& disp_operator(A other);
  #pragma omp declare variant(disp_operator)                            \
    match(construct={dispatch}, device={arch(arm)})
  A& operator=(A other);
};

struct Obj2 {
  A xx;
  Obj2& disp_operator(Obj2 other);
  #pragma omp declare variant(disp_operator)                            \
    match(construct={dispatch}, device={arch(arm)})
  Obj2& operator=(Obj2 other);

  void foo() {
    Obj2 z;
    //PRINT: #pragma omp dispatch
    //PRINT: z = z;
    //DUMP: OMPDispatchDirective
    #pragma omp dispatch
    z = z;
    //PRINT: #pragma omp dispatch
    //PRINT: z.operator=(z);
    //DUMP: OMPDispatchDirective
    #pragma omp dispatch
    z.operator=(z);
  }
  void bar() {
    Obj2 j;
    //PRINT: #pragma omp dispatch
    //PRINT: j = {this->xx};
    //DUMP: OMPDispatchDirective
    #pragma omp dispatch
    j = {this->xx};
    //PRINT: #pragma omp dispatch
    //PRINT: j.operator=({this->xx});
    //DUMP: OMPDispatchDirective
    #pragma omp dispatch
    j.operator=({this->xx});
  }
};

void test_three()
{
  Obj2 z1, z;
  #pragma omp dispatch
  z1 = z;
  #pragma omp dispatch
  z1.operator=(z);
}
#endif // HEADER
