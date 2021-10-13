//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -fsyntax-only -verify -o - %s

//RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                      \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                          \
//RUN:   -fsyntax-only -verify -o - %s

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -emit-pch -o %t %s

// expected-no-diagnostics

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
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

#endif // HEADER
