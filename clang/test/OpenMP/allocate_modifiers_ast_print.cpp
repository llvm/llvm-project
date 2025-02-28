// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++14 \
// RUN:   -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++14 \
// RUN:   -ast-dump %s | FileCheck %s --check-prefix=DUMP

// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++14 -emit-pch -o %t %s

// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++14 -include-pch \
// RUN:   %t -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++14 -include-pch \
// RUN:   %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// expected-no-diagnostics

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
} omp_allocator_handle_t;

omp_allocator_handle_t myAlloc() {
  return omp_large_cap_mem_alloc;
}

int main() {
  int a, b, c, d;
  #pragma omp scope private(a) allocate(omp_const_mem_alloc:a)
  a++;
  #pragma omp scope private(a,b) allocate(allocator(omp_const_mem_alloc):a,b)
  b++;
  #pragma omp scope private(c,a,b) allocate(allocator(myAlloc()):a,b,c)
  c++;
  #pragma omp scope private(c,a,b,d) allocate(myAlloc():a,b,c,d)
  a++;
  #pragma omp scope private(a,b) allocate(align(2), allocator(omp_const_mem_alloc):a,b)
  b++;
  #pragma omp scope private(c,a,b) allocate(allocator(myAlloc()), align(8) :a,b,c)
  c++;
// DUMP: FunctionDecl {{.*}}
// DUMP: DeclRefExpr {{.*}}'omp_allocator_handle_t' EnumConstant {{.*}}'omp_large_cap_mem_alloc' 'omp_allocator_handle_t'
// DUMP: FunctionDecl {{.*}}
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'d' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'d' 'int'
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPScopeDirective {{.*}}
// DUMP: OMPPrivateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: OMPAllocateClause {{.*}}
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'a' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'b' 'int'
// DUMP: DeclRefExpr {{.*}}'int' lvalue Var {{.*}}'c' 'int'
// PRINT: #pragma omp scope private(a) allocate(omp_const_mem_alloc: a)
// PRINT: #pragma omp scope private(a,b) allocate(allocator(omp_const_mem_alloc): a,b)
// PRINT: #pragma omp scope private(c,a,b) allocate(allocator(myAlloc()): a,b,c)
// PRINT: #pragma omp scope private(c,a,b,d) allocate(myAlloc(): a,b,c,d)
// PRINT: #pragma omp scope private(a,b) allocate(align(2), allocator(omp_const_mem_alloc): a,b)
// PRINT: #pragma omp scope private(c,a,b) allocate(allocator(myAlloc()), align(8): a,b,c)
  return a+b+c+d;
}

template<typename T, unsigned al>
void templated_func(T n) {
  int a, b;
  T mem = n;
  #pragma omp scope private(mem,a,b) allocate(allocator(n),align(al):mem,a,b)
  a += b;
  #pragma omp scope allocate(allocator(n),align(al):mem,a,b) private(mem,a,b)
  a += b;
}

void template_inst(int n) {
  templated_func<omp_allocator_handle_t, 4>(omp_const_mem_alloc);
  return;
}
// DUMP: FunctionTemplateDecl{{.*}}templated_func
// DUMP: FunctionDecl{{.*}}templated_func 'void (T)'
// DUMP: OMPScopeDirective
// DUMP: OMPPrivateClause
// DUMP: OMPAllocateClause
// DUMP: DeclRefExpr{{.*}}'T' lvalue Var{{.*}}'mem' 'T'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'a' 'int'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'b' 'int'
// DUMP: OMPScopeDirective
// DUMP: OMPAllocateClause
// DUMP: DeclRefExpr{{.*}}'T' lvalue Var{{.*}}'mem' 'T'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'a' 'int'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'b' 'int'
// DUMP: OMPPrivateClause

// DUMP: FunctionDecl{{.*}}used templated_func 'void (omp_allocator_handle_t)' implicit_instantiation
// DUMP: TemplateArgument type 'omp_allocator_handle_t'
// DUMP: EnumType{{.*}}'omp_allocator_handle_t'
// DUMP: Enum{{.*}}'omp_allocator_handle_t'
// DUMP: TemplateArgument integral '4U'

// DUMP: OMPScopeDirective
// DUMP: OMPPrivateClause
// DUMP: OMPAllocateClause
// DUMP: DeclRefExpr{{.*}}'omp_allocator_handle_t' lvalue Var{{.*}}'mem' 'omp_allocator_handle_t'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'a' 'int'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'b' 'int'
// DUMP: OMPScopeDirective
// DUMP: OMPAllocateClause
// DUMP: DeclRefExpr{{.*}}'omp_allocator_handle_t' lvalue Var{{.*}}'mem' 'omp_allocator_handle_t'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'a' 'int'
// DUMP: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'b' 'int'
// DUMP: OMPPrivateClause
// PRINT: #pragma omp scope private(mem,a,b) allocate(allocator(n), align(al): mem,a,b)
// PRINT: #pragma omp scope allocate(allocator(n), align(al): mem,a,b) private(mem,a,b)
// PRINT: #pragma omp scope private(mem,a,b) allocate(allocator(n), align(4U): mem,a,b)
// PRINT: #pragma omp scope allocate(allocator(n), align(4U): mem,a,b) private(mem,a,b)

#endif
