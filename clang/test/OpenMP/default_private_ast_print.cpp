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
//RUN:   -emit-pch -o %t %s

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

struct SomeKernel {
  int targetDev;
  float devPtr;
  SomeKernel();
  ~SomeKernel();

  template <unsigned int nRHS>
  void apply() {
#pragma omp parallel default(private)
    {
      targetDev++;
    }
    // PRINT: #pragma omp parallel default(private)
    // PRINT-NEXT: {
    // PRINT-NEXT:  this->targetDev++;
    // CHECK-NEXT: }
    // DUMP: -OMPParallelDirective
    // DUMP->NEXT: -OMPDefaultClause
  }
  // PRINT: template<> void apply<32U>()
  // PRINT: #pragma omp parallel default(private)
  // PRINT-NEXT: {
  // PRINT-NEXT:  this->targetDev++;
  // CHECK-NEXT: }
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT: -OMPDefaultClause
};

void use_template() {
  SomeKernel aKern;
  aKern.apply<32>();
}

void foo() {
  int a;
#pragma omp parallel default(private)
  a++;
  // PRINT: #pragma omp parallel default(private)
  // PRINT-NEXT: a++;
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP-NEXT: -OMPPrivateClause {{.*}} <implicit>
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'a'
}

struct St {
  int a, b;
  int y;
  St() : a(0), b(0) {}
  ~St() {}
};
void bar() {
  St a = St();
  static int yy = 0;
#pragma omp parallel default(private)
  {
    a.a += 1;
    a.b += 1;
    a.y++;
    yy++;
  }
  // PRINT: #pragma omp parallel default(private)
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPPrivateClause {{.*}} <implicit>
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'a'
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'yy'
}

void zoo(int);
struct A {
  int z;
  int f;
  A();
  ~A();
  void foo() {
#pragma omp parallel private(z) default(private)
    {
      z++;
      f++;
      zoo(z + f);
      f++;
    }
  }
  // PRINT:    #pragma omp parallel private(this->z) default(private)
  // DUMP:     -OMPParallelDirective
  // DUMP-NEXT:  -OMPPrivateClause
  // DUMP-NEXT:    -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP-NEXT:  -OMPPrivateClause
  // DUMP-NEXT:    -DeclRefExpr {{.*}} 'f'
  // DUMP:         -CXXThisExpr {{.*}} 'A *' implicit this
  void bar() {
#pragma omp parallel private(z) default(private)
    {
#pragma omp parallel private(z) default(private)
      {
        z++;
        f++;
        zoo(z + f);
        f++;
      }
    }
  }
  // PRINT:    #pragma omp parallel private(this->z) default(private)
  // PRINT:          #pragma omp parallel private(this->z) default(private)
  // DUMP:     -OMPParallelDirective
  // DUMP-NEXT:  -OMPPrivateClause
  // DUMP-NEXT:    -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP:           -OMPParallelDirective
  // DUMP-NEXT:        -OMPPrivateClause
  // DUMP-NEXT:           -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT:        -OMPDefaultClause
  // DUMP-NEXT:        -OMPPrivateClause {{.*}} <implicit>
  // DUMP-NEXT:           -DeclRefExpr {{.*}} 'f'
  // DUMP:             -CXXThisExpr
  // DUMP:         -MemberExpr
  // DUMP-NEXT:       -CXXThisExpr
  // DUMP:         -CXXThisExpr
};
#endif // HEADER
