// Test this without pch.
// RUN: %clang_cc1 -std=c++11 -fopenmp -fsyntax-only -verify %s -ast-dump-all | FileCheck %s -implicit-check-not=openmp_structured_block

// Test with pch. Use '-ast-dump' to force deserialization of function bodies.
// RUN: %clang_cc1 -std=c++11 -fopenmp -emit-pch -o %t %s
// RUN: echo "// expected-no-diagnostics" | %clang_cc1 -x c++ -std=c++11 -include-pch %t -fopenmp -fsyntax-only -verify - -ast-dump-all | FileCheck %s -implicit-check-not=openmp_structured_block

void test() {
#pragma omp parallel
  ;
}

// expected-no-diagnostics

// CHECK: TranslationUnitDecl 0x{{.*}} <<invalid sloc>> <invalid sloc>
// CHECK: `-FunctionDecl 0x{{.*}} <{{.*}}stmt-openmp_structured_block-bit.cpp:8:1, line:11:1> line:8:6 {{(test|imported test)}} 'void ()'
// CHECK-NEXT:   `-CompoundStmt 0x{{.*}} <col:13, line:11:1>
// CHECK-NEXT:     `-OMPParallelDirective 0x{{.*}} <line:9:1, col:21>
// CHECK-NEXT:       `-CapturedStmt 0x{{.*}} <line:10:3>
// CHECK-NEXT:         `-CapturedDecl 0x{{.*}} <<invalid sloc>> <invalid sloc> {{(nothrow|imported <undeserialized declarations> nothrow)}}
// CHECK-NEXT:           |-NullStmt 0x{{.*}} <col:3>
// CHECK-NEXT:           |-ImplicitParamDecl 0x{{.*}} <line:9:1> col:1 {{(implicit|imported implicit)}} .global_tid. 'const int *const __restrict'
// CHECK-NEXT:           |-ImplicitParamDecl 0x{{.*}} <col:1> col:1 {{(implicit|imported implicit)}} .bound_tid. 'const int *const __restrict'
// CHECK-NEXT:           `-ImplicitParamDecl 0x{{.*}} <col:1> col:1 {{(implicit|imported implicit)}} __context '(unnamed struct at {{.*}}stmt-openmp_structured_block-bit.cpp:9:1) *const __restrict'
