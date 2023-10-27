//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -fsyntax-only -verify -o - %s

//RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                      \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                          \
//RUN:   -fsyntax-only -verify -o - %s

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -emit-pch -o %t %s

// expected-no-diagnostics

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

void foo_v1(float *AAA, float *BBB, int *I) {return;}
void foo_v2(float *AAA, float *BBB, int *I) {return;}
void foo_v3(float *AAA, float *BBB, int *I) {return;}

//DUMP: FunctionDecl{{.*}} foo 'void (float *, float *, int *)'
//DUMP: OMPDeclareVariantAttr{{.*}}device={arch(x86, x86_64)}
//DUMP: DeclRefExpr{{.*}}Function{{.*}}foo_v3
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'I'
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'BBB'
//DUMP: OMPDeclareVariantAttr{{.*}}device={arch(ppc)}
//DUMP: DeclRefExpr{{.*}}Function{{.*}}foo_v2
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'AAA'
//DUMP: OMPDeclareVariantAttr{{.*}}device={arch(arm)}
//DUMP: DeclRefExpr{{.*}}Function{{.*}}foo_v1
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'AAA'
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'BBB'
//PRINT: #pragma omp declare variant(foo_v3) match(construct={dispatch}, device={arch(x86, x86_64)}) adjust_args(nothing:I) adjust_args(need_device_ptr:BBB)

//PRINT: #pragma omp declare variant(foo_v2) match(construct={dispatch}, device={arch(ppc)}) adjust_args(need_device_ptr:AAA)

//PRINT: omp declare variant(foo_v1) match(construct={dispatch}, device={arch(arm)}) adjust_args(need_device_ptr:AAA,BBB)

#pragma omp declare variant(foo_v1)                        \
   match(construct={dispatch}, device={arch(arm)})         \
   adjust_args(need_device_ptr:AAA,BBB)

#pragma omp declare variant(foo_v2)                        \
   match(construct={dispatch}, device={arch(ppc)}),        \
   adjust_args(need_device_ptr:AAA)

#pragma omp declare variant(foo_v3)                        \
   adjust_args(need_device_ptr:BBB) adjust_args(nothing:I) \
   match(construct={dispatch}, device={arch(x86,x86_64)})

void foo(float *AAA, float *BBB, int *I) {return;}

void Foo_Var(float *AAA, float *BBB) {return;}

#pragma omp declare variant(Foo_Var) \
   match(construct={dispatch}, device={arch(x86_64)}) \
   adjust_args(need_device_ptr:AAA) adjust_args(nothing:BBB)
template<typename T>
void Foo(T *AAA, T *BBB) {return;}

//PRINT: #pragma omp declare variant(Foo_Var) match(construct={dispatch}, device={arch(x86_64)}) adjust_args(nothing:BBB) adjust_args(need_device_ptr:AAA)
//DUMP: FunctionDecl{{.*}} Foo 'void (T *, T *)'
//DUMP: OMPDeclareVariantAttr{{.*}}device={arch(x86_64)}
//DUMP: DeclRefExpr{{.*}}Function{{.*}}Foo_Var
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'BBB'
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'AAA'
//
//DUMP: FunctionDecl{{.*}} Foo 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}device={arch(x86_64)}
//DUMP: DeclRefExpr{{.*}}Function{{.*}}Foo_Var
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'BBB'
//DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'AAA'

void func()
{
  float *A;
  float *B;

  //#pragma omp dispatch
  Foo(A, B);
}

typedef void *omp_interop_t;

void bar_v1(float* F1, float *F2, omp_interop_t);
void bar_v2(float* F1, float *F2, omp_interop_t, omp_interop_t);

//PRINT: #pragma omp declare variant(bar_v1) match(construct={dispatch}) append_args(interop(target,targetsync))
//DUMP: FunctionDecl{{.*}}bar1 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}construct={dispatch} Target_TargetSync
//DUMP: DeclRefExpr{{.*}}bar_v1
#pragma omp declare variant(bar_v1) match(construct={dispatch}) \
                                    append_args(interop(target,targetsync))
void bar1(float *FF1, float *FF2) { return; }

//PRINT: #pragma omp declare variant(bar_v1) match(construct={dispatch}) append_args(interop(targetsync))
//DUMP: FunctionDecl{{.*}}bar2 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}construct={dispatch} TargetSync
//DUMP: DeclRefExpr{{.*}}bar_v1
#pragma omp declare variant(bar_v1) match(construct={dispatch}) \
                                    append_args(interop(targetsync))
void bar2(float *FF1, float *FF2) { return; }

//PRINT: #pragma omp declare variant(bar_v1) match(construct={dispatch}) append_args(interop(target))
//DUMP: FunctionDecl{{.*}}bar3 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}construct={dispatch} Target
//DUMP: DeclRefExpr{{.*}}bar_v1
#pragma omp declare variant(bar_v1) match(construct={dispatch}) \
                                    append_args(interop(target))
void bar3(float *FF1, float *FF2) { return; }

//PRINT: #pragma omp declare variant(bar_v2) match(construct={dispatch}) append_args(interop(target), interop(targetsync))
//DUMP: FunctionDecl{{.*}}bar4 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}construct={dispatch} Target TargetSync
//DUMP: DeclRefExpr{{.*}}bar_v2
#pragma omp declare variant(bar_v2) match(construct={dispatch}) \
                       append_args(interop(target), interop(targetsync))
void bar4(float *FF1, float *FF2) { return; }

//PRINT: #pragma omp declare variant(bar_v2) match(construct={dispatch}) append_args(interop(targetsync), interop(target))
//DUMP: FunctionDecl{{.*}}bar5 'void (float *, float *)'
//DUMP: OMPDeclareVariantAttr{{.*}}construct={dispatch} TargetSync Target
//DUMP: DeclRefExpr{{.*}}bar_v2
#pragma omp declare variant(bar_v2) match(construct={dispatch}) \
                       append_args(interop(targetsync), interop(target))
void bar5(float *FF1, float *FF2) { return; }

//PRINT: class A {
//DUMP: CXXRecordDecl{{.*}}class A definition
class A {
public:
  void memberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp);
  //PRINT: #pragma omp declare variant(memberfoo_v1) match(construct={dispatch}) append_args(interop(target))
  //DUMP: CXXMethodDecl{{.*}}memberbar 'void (float *, float *, int *)'
  //DUMP: OMPDeclareVariantAttr{{.*}}Implicit construct={dispatch} Target
  //DUMP: DeclRefExpr{{.*}}'memberfoo_v1' 'void (float *, float *, int *, omp_interop_t)'
  #pragma omp declare variant(memberfoo_v1) match(construct={dispatch}) \
    append_args(interop(target))
  void memberbar(float *A, float *B, int *I) { return; }

  static void smemberfoo_v1(float *A, float *B, int *I, omp_interop_t IOp);
  //PRINT: #pragma omp declare variant(smemberfoo_v1) match(construct={dispatch}) append_args(interop(target))
  //DUMP: CXXMethodDecl{{.*}}smemberbar 'void (float *, float *, int *)' static
  //DUMP: OMPDeclareVariantAttr{{.*}}Implicit construct={dispatch} Target
  //DUMP: DeclRefExpr{{.*}}'smemberfoo_v1' 'void (float *, float *, int *, omp_interop_t)'
  #pragma omp declare variant(smemberfoo_v1) match(construct={dispatch}) \
    append_args(interop(target))
  static void smemberbar(float *A, float *B, int *I) { return; }
};

template <typename T> void templatefoo_v1(const T& t, omp_interop_t I);
template <typename T> void templatebar(const T& t) {}

//PRINT: #pragma omp declare variant(templatefoo_v1<int>) match(construct={dispatch}) append_args(interop(target))
//DUMP: FunctionDecl{{.*}}templatebar 'void (const int &)'
//DUMP: OMPDeclareVariantAttr{{.*}}Implicit construct={dispatch} Target
//DUMP: DeclRefExpr{{.*}}'templatefoo_v1' 'void (const int &, omp_interop_t)'
#pragma omp declare variant(templatefoo_v1<int>) match(construct={dispatch}) \
  append_args(interop(target))
void templatebar(const int &t) {}
#endif // HEADER
