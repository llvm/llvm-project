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

typedef void *omp_interop_t;

//PRINT-LABEL: void foo1(
//DUMP-LABEL:  FunctionDecl {{.*}} foo1
void foo1(int *ap, int dev) {
  omp_interop_t I;
  omp_interop_t &IRef = I;

  //PRINT: #pragma omp interop init(target : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  #pragma omp interop init(target:I)

  //PRINT: #pragma omp interop init(target : IRef)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'IRef'
  #pragma omp interop init(target:IRef)

  //PRINT: #pragma omp interop device(dev) depend(inout : ap) init(targetsync : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPDeviceClause
  //DUMP: DeclRefExpr{{.*}}'dev' 'int'
  //DUMP: OMPDependClause
  //DUMP: DeclRefExpr{{.*}}'ap' 'int *'
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  #pragma omp interop device(dev) depend(inout:ap) init(targetsync:I)

  //PRINT: #pragma omp interop init(prefer_type(1,2,3,4,5,6), targetsync : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  //DUMP: IntegerLiteral{{.*}}1
  //DUMP: IntegerLiteral{{.*}}2
  //DUMP: IntegerLiteral{{.*}}3
  //DUMP: IntegerLiteral{{.*}}4
  //DUMP: IntegerLiteral{{.*}}5
  //DUMP: IntegerLiteral{{.*}}6
  #pragma omp interop init(prefer_type(1,2,3,4,5,6),targetsync:I)

  //PRINT: #pragma omp interop init(prefer_type(2,4,6,1), targetsync : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  //DUMP: IntegerLiteral{{.*}}2
  //DUMP: IntegerLiteral{{.*}}4
  //DUMP: IntegerLiteral{{.*}}6
  //DUMP: IntegerLiteral{{.*}}1
  #pragma omp interop init(prefer_type(2,4,6,1),targetsync:I)

  //PRINT: #pragma omp interop init(prefer_type("cuda","cuda_driver","opencl","sycl","hip","level_zero"), targetsync : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  //DUMP: StringLiteral{{.*}}"cuda"
  //DUMP: StringLiteral{{.*}}"cuda_driver"
  //DUMP: StringLiteral{{.*}}"opencl"
  //DUMP: StringLiteral{{.*}}"sycl"
  //DUMP: StringLiteral{{.*}}"hip"
  //DUMP: StringLiteral{{.*}}"level_zero"
  #pragma omp interop init( \
    prefer_type("cuda","cuda_driver","opencl","sycl","hip","level_zero"), \
    targetsync:I)

  //PRINT: #pragma omp interop init(prefer_type("level_zero",2,4), targetsync : I)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  //DUMP: StringLiteral{{.*}}"level_zero"
  //DUMP: IntegerLiteral{{.*}}2
  //DUMP: IntegerLiteral{{.*}}4
  #pragma omp interop init(prefer_type("level_zero",2,4),targetsync:I)

  omp_interop_t J;

  //PRINT: #pragma omp interop init(target : I) init(targetsync : J)
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'I'
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}Var{{.*}}'J'
  #pragma omp interop init(target:I) init(targetsync:J)

}

//DUMP: FunctionTemplateDecl{{.*}}fooTemp
//DUMP-NEXT: NonTypeTemplateParmDecl{{.*}}'int{{.*}}I
template <int I>
void fooTemp() {
  omp_interop_t interop_var;
  //PRINT: #pragma omp interop init(prefer_type(I,4,"level_one"), target : interop_var)
  //DUMP: FunctionDecl{{.*}}fooTemp
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}'interop_var'
  //DUMP: DeclRefExpr{{.*}}NonTypeTemplateParm{{.*}}'I' 'int'
  //DUMP: IntegerLiteral{{.*}}'int' 4
  //DUMP: StringLiteral{{.*}}"level_one"

  //PRINT: #pragma omp interop init(prefer_type(3,4,"level_one"), target : interop_var)
  //DUMP: FunctionDecl{{.*}}fooTemp
  //DUMP: TemplateArgument integral 3
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}'omp_interop_t'{{.*}}'interop_var'
  //DUMP: SubstNonTypeTemplateParmExpr{{.*}}'int'
  //DUMP: NonTypeTemplateParmDecl{{.*}}'int'{{.*}}I
  //DUMP: IntegerLiteral{{.*}}'int' 3
  //DUMP: IntegerLiteral{{.*}}'int' 4
  //DUMP: StringLiteral{{.*}}"level_one"
  #pragma omp interop init(prefer_type(I,4,"level_one"), target: interop_var)
}

//DUMP: FunctionTemplateDecl{{.*}}barTemp
//DUMP-NEXT: TemplateTypeParmDecl{{.*}}typename{{.*}}T
template <typename T>
void barTemp(T t) {
  //PRINT: #pragma omp interop init(prefer_type(4,"level_one"), target : t)
  //DUMP: FunctionDecl{{.*}}barTemp 'void (T)'
  //DUMP: ParmVarDecl{{.*}}t 'T'
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'t' 'T'
  //DUMP: IntegerLiteral{{.*}}'int' 4
  //DUMP: StringLiteral{{.*}}"level_one"
  #pragma omp interop init(prefer_type(4,"level_one"), target: t)

  //DUMP: FunctionDecl{{.*}}barTemp 'void (void *)'
  //DUMP: TemplateArgument type 'void *'
  //DUMP: ParmVarDecl{{.*}}t 'void *'
  //DUMP: OMPInteropDirective
  //DUMP: OMPInitClause
  //DUMP: DeclRefExpr{{.*}}ParmVar{{.*}}'t' 'void *'
  //PRINT: #pragma omp interop init(prefer_type(4,"level_one"), target : t)
}

void bar()
{
  fooTemp<3>();
  omp_interop_t Ivar;
  barTemp(Ivar);
}

#endif // HEADER
