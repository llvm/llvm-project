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

typedef enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
} omp_allocator_handle_t;

int foo1() {
  char a;
#pragma omp allocate(a) align(4) allocator(omp_pteam_mem_alloc)
  return a;
}
// DUMP: FunctionDecl {{.*}}
// DUMP: DeclStmt {{.*}}
// DUMP: VarDecl {{.*}}a 'char'
// DUMP: OMPAllocateDeclAttr {{.*}}OMPPTeamMemAlloc
// DUMP: DeclRefExpr {{.*}}'omp_allocator_handle_t' EnumConstant {{.*}} 'omp_pteam_mem_alloc' 'omp_allocator_handle_t'
// DUMP: ConstantExpr {{.*}}'int'
// DUMP: value: Int 4
// DUMP: IntegerLiteral {{.*}}'int' 4
// DUMP: DeclStmt {{.*}}
// DUMP: OMPAllocateDecl {{.*}}
// DUMP: DeclRefExpr {{.*}}'char' lvalue Var {{.*}} 'a' 'char'
// DUMP: OMPAlignClause {{.*}}
// DUMP: ConstantExpr {{.*}}'int'
// DUMP: value: Int 4
// DUMP: IntegerLiteral {{.*}}'int' 4
// DUMP: OMPAllocatorClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'omp_allocator_handle_t' EnumConstant {{.*}}'omp_pteam_mem_alloc' 'omp_allocator_handle_t'
// PRINT: #pragma omp allocate(a) align(4) allocator(omp_pteam_mem_alloc)

int foo2() {
  char b;
#pragma omp allocate(b) allocator(omp_low_lat_mem_alloc) align(2)
  return b;
}
// DUMP: FunctionDecl {{.*}}
// DUMP: DeclStmt {{.*}}
// DUMP: VarDecl {{.*}}b 'char'
// DUMP: OMPAllocateDeclAttr {{.*}}Implicit OMPLowLatMemAlloc
// DUMP: DeclRefExpr {{.*}}'omp_allocator_handle_t' EnumConstant {{.*}} 'omp_low_lat_mem_alloc' 'omp_allocator_handle_t'
// DUMP: ConstantExpr {{.*}}'int'
// DUMP: value: Int 2
// DUMP: IntegerLiteral {{.*}}'int' 2
// DUMP: DeclStmt {{.*}}
// DUMP: OMPAllocateDecl {{.*}}
// DUMP: DeclRefExpr {{.*}}'char' lvalue Var {{.*}} 'b' 'char'
// DUMP: OMPAllocatorClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'omp_allocator_handle_t' EnumConstant {{.*}} 'omp_low_lat_mem_alloc' 'omp_allocator_handle_t'
// DUMP: OMPAlignClause {{.*}}
// DUMP:  ConstantExpr {{.*}}'int'
// DUMP: value: Int 2
// DUMP: IntegerLiteral {{.*}}'int' 2
// PRINT: #pragma omp allocate(b) allocator(omp_low_lat_mem_alloc) align(2)

template <typename T, unsigned size>
T run() {
  T foo;
#pragma omp allocate(foo) align(size)
  return size;
}

int template_test() {
  double d;
  d = run<double, 1>();
  return 0;
}

// DUMP: FunctionTemplateDecl {{.*}}
// DUMP: TemplateTypeParmDecl {{.*}}
// DUMP: NonTypeTemplateParmDecl {{.*}}'unsigned int' depth 0 index 1 size
// DUMP: FunctionDecl {{.*}}'T ()'
// DUMP: DeclStmt {{.*}}
// DUMP: OMPAllocateDecl {{.*}}
// DUMP: DeclRefExpr {{.*}}'T' lvalue Var {{.*}} 'foo' 'T'
// DUMP: OMPAlignClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'unsigned int' NonTypeTemplateParm {{.*}} 'size' 'unsigned int'
// DUMP: FunctionDecl {{.*}}run 'double ()'
// DUMP: TemplateArgument type 'double'
// DUMP: BuiltinType {{.*}}'double'
// DUMP: TemplateArgument integral '1U'
// DUMP: OMPAllocateDeclAttr {{.*}}Implicit OMPNullMemAlloc
// DUMP: ConstantExpr {{.*}}'unsigned int'
// DUMP: value: Int 1
// DUMP: SubstNonTypeTemplateParmExpr {{.*}}'unsigned int'
// DUMP: NonTypeTemplateParmDecl {{.*}}'unsigned int' depth 0 index 1 size
// DUMP: IntegerLiteral {{.*}}'unsigned int' 1
// DUMP: OMPAllocateDecl {{.*}}
// DUMP: DeclRefExpr {{.*}}'double' lvalue Var {{.*}} 'foo' 'double'
// DUMP: OMPAlignClause {{.*}}
// DUMP: ConstantExpr {{.*}}'unsigned int'
// DUMP: value: Int 1
// DUMP: SubstNonTypeTemplateParmExpr {{.*}}'unsigned int'
// DUMP: NonTypeTemplateParmDecl {{.*}}'unsigned int' depth 0 index 1 size
// DUMP: IntegerLiteral {{.*}}'unsigned int' 1
// PRINT: #pragma omp allocate(foo) align(size)
// PRINT: #pragma omp allocate(foo) align(1U)
#endif // HEADER
