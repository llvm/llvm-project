// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=c_mode         -ast-dump %s                      | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=cxx_mode       -ast-dump %s -x c++ %std_cxx11-14 | FileCheck %s --check-prefixes=CHECK,CXX
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=cxx_mode,cxx17 -ast-dump %s -x c++ %std_cxx17-   | FileCheck %s --check-prefixes=CHECK,CXX
// c_mode-no-diagnostics

#ifdef __cplusplus
#define CONST constexpr
#else
#define CONST __attribute__((const))
#endif

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
CONST int also_after1() { // cxx_mode-note {{previous declaration is here}}
  return 0;
}
static int also_after2() {
  return 0;
}
__attribute__((nothrow)) int also_after3() { // cxx17-note {{previous declaration is here}}
  return 0;
}
static CONST __attribute__((nothrow, always_inline)) __inline__ int also_after4() { // cxx_mode-note {{previous declaration is here}} cxx17-note {{previous declaration is here}}
  return 0;
}
#pragma omp end declare variant

int also_after1() { // cxx_mode-error {{non-constexpr declaration of 'also_after1' follows constexpr declaration}}
  return 1;
}
int also_after2() {
  return 2;
}
int also_after3() { // cxx17-warning {{'also_after3' is missing exception specification '__attribute__((nothrow))'}}
  return 3;
}
int also_after4() { // cxx_mode-error {{non-constexpr declaration of 'also_after4' follows constexpr declaration}} cxx17-warning {{'also_after4' is missing exception specification '__attribute__((nothrow))'}}
  return 4;
}


int main(void) {
  // Should return 0.
  return also_after1() + also_after2() + also_after3() + also_after4();
}

// Make sure:
//  - the right FunctionDecls are marked as used

// CHECK:      FunctionDecl {{.*}} used{{.*}} also_after1 'int ()'
// CHECK:        OMPDeclareVariantAttr {{.*}} Implicit implementation={vendor(llvm)}
// CHECK-NEXT:     DeclRefExpr {{.*}} 'also_after1[implementation={vendor(llvm)}]'
// CHECK-NEXT: FunctionDecl {{.*}} also_after1[implementation={vendor(llvm)}]

// CHECK:      FunctionDecl {{.*}} used also_after2 'int ()'
// CHECK-NEXT:   OMPDeclareVariantAttr {{.*}} Implicit implementation={vendor(llvm)}
// CHECK-NEXT:     DeclRefExpr {{.*}} 'also_after2[implementation={vendor(llvm)}]'
// CHECK-NEXT: FunctionDecl {{.*}} also_after2[implementation={vendor(llvm)}]

// C:          FunctionDecl {{.*}} used also_after3 'int ()'
// CXX:        FunctionDecl {{.*}} used also_after3 'int () __attribute__((nothrow))'
// CHECK:        OMPDeclareVariantAttr {{.*}} Implicit implementation={vendor(llvm)}
// CHECK-NEXT:     DeclRefExpr {{.*}} 'also_after3[implementation={vendor(llvm)}]'
// CHECK-NEXT: FunctionDecl {{.*}} also_after3[implementation={vendor(llvm)}]

// C:          FunctionDecl {{.*}} used{{.*}} also_after4 'int ()'
// CXX:        FunctionDecl {{.*}} used{{.*}} also_after4 'int () __attribute__((nothrow))'
// CHECK:        OMPDeclareVariantAttr {{.*}} Implicit implementation={vendor(llvm)}
// CHECK-NEXT:     DeclRefExpr {{.*}} 'also_after4[implementation={vendor(llvm)}]'
// CHECK-NEXT: FunctionDecl {{.*}} also_after4[implementation={vendor(llvm)}]
