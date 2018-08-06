// RUN: %clang_cc1 -triple i686--windows-msvc -fexceptions -fobjc-exceptions -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64--windows-msvc -fexceptions -fobjc-exceptions -ast-dump %s 2>&1 | FileCheck %s

void f() {
  @try {
  } @finally {
  }
}

// CHECK:      ObjCAtFinallyStmt
// CHECK-NEXT:   CapturedStmt
// CHECK-NEXT:     CapturedDecl
// CHECK-NEXT:       CompoundStmt
// CHECK-NEXT:       ImplicitParamDecl
