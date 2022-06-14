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
//RUN:   -emit-pch -o %t %s

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

struct SomeKernel {
  int targetDev;
  float devPtr;
  SomeKernel();
  ~SomeKernel();

  template <unsigned int nRHS>
  void apply() {
#pragma omp parallel default(firstprivate)
    {
      [=]() -> int {
        return targetDev++;
      }();
    }
    // PRINT: #pragma omp parallel default(firstprivate)
    // PRINT-NEXT: {
    // PRINT-NEXT:  [=]() -> int {
    // PRINT-NEXT:     return this->targetDev++;
    // PRINT-NEXT:  }();
    // PRINT-NEXT: }
    // DUMP: -OMPParallelDirective
    // DUMP-NEXT: -OMPDefaultClause
    // DUMP-NOT:   -OMPFirstprivateClause
  }
  // PRINT: template<> void apply<32U>()
  // PRINT: #pragma omp parallel default(firstprivate)
  // PRINT-NEXT: {
  // PRINT-NEXT:  [=]() -> int {
  // PRINT-NEXT:     return this->targetDev++;
  // PRINT-NEXT:  }();
  // CHECK-NEXT: }
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPFirstprivateClause
  // DUMP-NEXT:   -DeclRefExpr {{.*}} 'targetDev'
};

void use_template() {
  SomeKernel aKern;
  aKern.apply<32>();
}

void foo() {
  int a;
#pragma omp parallel default(firstprivate)
  a++;
  // PRINT: #pragma omp parallel default(firstprivate)
  // PRINT-NEXT: a++;
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP-NEXT:  -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT:   -DeclRefExpr {{.*}} 'a'
}

struct St {
  int a, b;
  static int y;
  St() : a(0), b(0) {}
  ~St() {}
};
int St::y = 0;
void bar() {
  St a = St();
  static int yy = 0;
#pragma omp parallel default(firstprivate)
  {
    a.a += 1;
    a.b += 1;
    a.y++;
    yy++;
    St::y++;
  }
  // PRINT: #pragma omp parallel default(firstprivate)
  // DUMP: -OMPParallelDirective
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'a'
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'yy'
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'y'
}
void zoo(int);
struct A {
  int z;
  int f;
  A();
  ~A();
  void foo() {
#pragma omp parallel firstprivate(z) default(firstprivate)
    {
      z++;
      f++;
      zoo(z + f);
      f++;
    }
  }
  // PRINT:  #pragma omp parallel firstprivate(this->z) default(firstprivate)
  // DUMP:   -OMPParallelDirective
  // DUMP-NEXT: -OMPFirstprivateClause
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT: -OMPDefaultClause
  // DUMP-NEXT: -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'f'
  // DUMP:      -CXXThisExpr {{.*}} 'A *' implicit this
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT: -DeclRefExpr {{.*}} 'f'
  void bar() {
#pragma omp parallel firstprivate(z) default(firstprivate)
    {
#pragma omp parallel private(z) default(firstprivate)
      {
        z++;
        f++;
        zoo(z + f);
        f++;
      }
    }
  }
  // PRINT:  #pragma omp parallel firstprivate(this->z) default(firstprivate)
  // PRINT:    #pragma omp parallel private(this->z) default(firstprivate)
  // DUMP:     -OMPParallelDirective
  // DUMP-NEXT: -OMPFirstprivateClause
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT:  -OMPDefaultClause
  // DUMP:        -OMPParallelDirective
  // DUMP-NEXT:    -OMPPrivateClaus
  // DUMP-NEXT:     -DeclRefExpr {{.*}} 'z'
  // DUMP-NEXT:     -OMPDefaultClause
  // DUMP-NEXT:     -OMPFirstprivateClause {{.*}} <implicit>
  // DUMP-NEXT:      -DeclRefExpr {{.*}} 'f'
  // DUMP:           -CXXThisExpr {{.*}} 'A *' implicit this
  // DUMP-NEXT:      -DeclRefExpr {{.*}} 'f'
  // DUMP:         -MemberExpr {{.*}}
  // DUMP-NEXT:      -CXXThisExpr
  // DUMP:       -CXXThisExpr {{.*}} 'A *' implicit this
  // DUMP-NEXT:  -DeclRefExpr {{.*}} 'z'
};
#endif // HEADER
