// expected-no-diagnostics

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -emit-pch -o %t %s

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

void foo() {
  int a;
#pragma omp target default(firstprivate)
  a++;
  // PRINT: #pragma omp target default(firstprivate)
  // PRINT-NEXT: a++;
  // DUMP: -OMPTargetDirective
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP-NEXT:  -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT:   -DeclRefExpr {{.*}} 'a'

}
void fun(){
int a = 0;
    int x = 10;
    #pragma omp target data default(firstprivate) map(a)
    {
  // DUMP: -OMPTargetDataDirective
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPMapClause
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'a'
  // DUMP-NEXT: -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'x'


        x += 10;
        a += 1;
    }
}
void bar(){
int i = 0;
int j = 0;
int  nn = 10;
#pragma omp target default(firstprivate)
#pragma omp teams 
#pragma teams distribute parallel for simd 
        for (j = 0; j < nn; j++ ) {
          for (i = 0; i < nn; i++ ) {
                ;
          }
        }

  // PRINT: #pragma omp target default(firstprivate)
  // DUMP: -OMPTargetDirective
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'j'
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'nn'
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'i'
}
#endif
