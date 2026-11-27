//RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -ast-dump  %s | FileCheck %s --check-prefix=DUM

typedef struct {
  int a;
} C;
#pragma omp declare mapper(C s) map(to : s.a)

typedef struct {
  int e;
  C f;
  int h;
} D;

void foo() {
  D sa[10];
  sa[1].e = 111;
  sa[1].f.a = 222;

#pragma omp target map(tofrom : sa [0:2])
  {
    sa[0].e = 333;
    sa[1].f.a = 444;
  }
}

// DUM: -OMPDeclareMapperDecl{{.*}}<<invalid sloc>> <invalid sloc>
// DUM-NEXT:  |-OMPMapClause {{.*}}<<invalid sloc>> <implicit>
// DUM-NEXT:  | |-MemberExpr {{.*}}<line:9:3> 'int' lvalue .e
// DUM-NEXT:  | | `-DeclRefExpr {{.*}}<<invalid sloc>> 'D' lvalue Var {{.*}} '_s' 'D'
// DUM-NEXT:  | |-MemberExpr {{.*}}<line:10:3> 'C':'struct C' lvalue .f {{.*}}
// DUM-NEXT:  | | `-DeclRefExpr {{.*}}<<invalid sloc>> 'D' lvalue Var {{.*}} '_s' 'D'
// DUM-NEXT:  | `-MemberExpr {{.*}}<line:11:3> 'int' lvalue .h {{.*}}
// DUM-NEXT:  |   `-DeclRefExpr {{.*}}<<invalid sloc>> 'D' lvalue Var {{.*}} '_s' 'D'
// DUM-NEXT:  `-VarDecl {{.*}} <line:12:1> col:1 implicit used _s 'D'
