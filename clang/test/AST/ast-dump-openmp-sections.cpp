// RUN: %clang_cc1 -verify -fopenmp -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -ast-dump %s | FileCheck %s
// expected-no-diagnostics

void sections() {
#pragma omp sections
  {
#pragma omp section
    {
    }
#pragma omp section
    {
    }
  }
}

// CHECK: `-FunctionDecl
// CHECK-NEXT:   `-CompoundStmt
// CHECK-NEXT:     `-OMPSectionsDirective
// CHECK-NEXT:       `-CapturedStmt
// CHECK-NEXT:         `-CapturedDecl {{.*}} nothrow
// CHECK-NEXT:           |-CompoundStmt
// CHECK-NEXT:           | |-OMPSectionDirective
// CHECK-NEXT:           | | `-CapturedStmt
// CHECK-NEXT:           | |   `-CapturedDecl {{.*}} nothrow
// CHECK-NEXT:           | |     |-CompoundStmt
// CHECK-NEXT:           | |     `-ImplicitParamDecl
// CHECK-NEXT:           | `-OMPSectionDirective
// CHECK-NEXT:           |   `-CapturedStmt
// CHECK-NEXT:           |     `-CapturedDecl {{.*}} nothrow
// CHECK-NEXT:           |       |-CompoundStmt
// CHECK-NEXT:           |       `-ImplicitParamDecl
// CHECK-NEXT:           |-ImplicitParamDecl
// CHECK-NEXT:           |-CXXRecordDecl
// CHECK-NEXT:           | |-DefinitionData
// CHECK-NEXT:           | | |-DefaultConstructor
// CHECK-NEXT:           | | |-CopyConstructor
// CHECK-NEXT:           | | |-MoveConstructor
// CHECK-NEXT:           | | |-CopyAssignment
// CHECK-NEXT:           | | |-MoveAssignment
// CHECK-NEXT:           | | `-Destructor
// CHECK-NEXT:           | `-CapturedRecordAttr
// CHECK-NEXT:           |-CapturedDecl {{.*}} nothrow
// CHECK-NEXT:           | |-CompoundStmt
// CHECK-NEXT:           | `-ImplicitParamDecl
// CHECK-NEXT:           |-CXXRecordDecl
// CHECK-NEXT:           | |-DefinitionData
// CHECK-NEXT:           | | |-DefaultConstructor
// CHECK-NEXT:           | | |-CopyConstructor
// CHECK-NEXT:           | | |-MoveConstructor
// CHECK-NEXT:           | | |-CopyAssignment
// CHECK-NEXT:           | | |-MoveAssignment
// CHECK-NEXT:           | | `-Destructor
// CHECK-NEXT:           | `-CapturedRecordAttr
// CHECK-NEXT:           `-CapturedDecl {{.*}} nothrow
// CHECK-NEXT:             |-CompoundStmt
// CHECK-NEXT:             `-ImplicitParamDecl
