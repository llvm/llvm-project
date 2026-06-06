//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu\
//RUN:   -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu\
//RUN:   -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu\
//RUN:   -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -emit-pch -o %t %s

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu\
//RUN:   -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT

//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu\
//RUN:   -fopenmp -fopenmp-version=51 \
//RUN:   -x c++ -std=c++14 -fexceptions -fcxx-exceptions                   \
//RUN:   -Wno-source-uses-openmp -Wno-openmp-clauses                       \
//RUN:   -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER
int foo1() {
  int a;
  int i = 1;
  #pragma omp scope private(a) reduction(+:i) nowait
  { 
    a = 123; 
    ++i; 
  }
  return i;
}

//DUMP: FunctionDecl {{.*}}foo1 'int ()'
//DUMP: OMPScopeDirective
//DUMP: OMPPrivateClause
//DUMP: DeclRefExpr {{.*}}'int' lvalue Var{{.*}}'a' 'int'
//DUMP: OMPReductionClause
//DUMP: DeclRefExpr {{.*}}'int' lvalue Var{{.*}}'i' 'int'
//DUMP: OMPNowaitClause
//PRINT: #pragma omp scope private(a) reduction(+: i) nowait

template <typename T>
T run() {
  T a;
  T b;

  #pragma omp scope private(a) reduction(*:b)
  { 
    b *= a; 
  }
  return b;
}

int template_test() {
  double d;
  d = run<double>();
  return 0;
}

//DUMP: FunctionTemplateDecl {{.*}}run
//DUMP: TemplateTypeParmDecl {{.*}}referenced typename depth 0 index 0 T
//DUMP: FunctionDecl {{.*}}run 'T ()'
//DUMP: OMPScopeDirective
//DUMP: OMPPrivateClause
//DUMP: DeclRefExpr {{.*}}'T' lvalue Var {{.*}} 'a' 'T'
//DUMP: OMPReductionClause
//DUMP: DeclRefExpr {{.*}}'T' lvalue Var {{.*}} 'b' 'T'
//DUMP: FunctionDecl {{.*}}used run 'double ()'
//DUMP: TemplateArgument type 'double'
//DUMP: BuiltinType {{.*}}'double'
//DUMP: OMPScopeDirective
//DUMP: OMPPrivateClause
//DUMP: DeclRefExpr {{.*}}'double' lvalue Var {{.*}} 'a' 'double'
//DUMP: OMPReductionClause
//DUMP: DeclRefExpr {{.*}}'double' lvalue Var {{.*}} 'b' 'double'
//PRINT: #pragma omp scope private(a) reduction(*: b)
#endif // HEADER
