// RUN: %clang_cc1 %s -verify -fopenacc
// RUN: not %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

void foo(int Var) {
  // expected-warning@+1{{unsupported OpenACC extension clause '__extension'}}
#pragma acc parallel copy(Var) __extension copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: NullStmt

  // expected-warning@+1{{unsupported OpenACC extension clause '__extension'}}
#pragma acc parallel copy(Var) __extension(stuff) copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: NullStmt

  // expected-warning@+1{{unsupported OpenACC extension clause '__extension'}}
#pragma acc parallel copy(Var) __extension(")") copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: NullStmt

  // expected-warning@+1{{unsupported OpenACC extension clause '__extension'}}
#pragma acc parallel copy(Var) __extension(()) copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: NullStmt

  // expected-warning@+2{{unsupported OpenACC extension clause '__extension'}}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel copy(Var) __extension()) copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // Cannot recover from a bad paren, so we give up here.
  // CHECK-NEXT: NullStmt

  // expected-warning@+3{{unsupported OpenACC extension clause '__extension'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel copy(Var) __extension(() copyin(Var)
  ;
  // CHECK: OpenACCComputeConstruct
  // CHECK-NEXT: copy clause
  // CHECK-NEXT: DeclRefExpr
  // Cannot recover from a bad paren, so we give up here.
  // CHECK-NEXT: NullStmt

}
