// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=C   -ast-dump %s       | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify=CXX -ast-dump %s -x c++| FileCheck %s --check-prefixes=CHECK,CXX


int OK_1(void);

#pragma omp begin declare variant match(implementation={vendor(intel)})
int OK_1(void) {
  return 1;
}
int OK_2(void) {
  return 1;
}
int not_OK(void) {
  return 1;
}
int OK_3(void) {
  return 1;
}
#pragma omp end declare variant

int OK_3(void);

int test(void) {
  // Should cause an error due to not_OK()
  return OK_1() + not_OK() + OK_3(); // CXX-error {{use of undeclared identifier 'not_OK'}} C-error {{call to undeclared function 'not_OK'; ISO C99 and later do not support implicit function declarations}}
}

// Make sure:
//  - we see a single error for `not_OK`
//  - we do not see errors for OK_{1,2,3}

// CHECK:      |-FunctionDecl [[ADDR_0:0x[a-z0-9]*]] <{{.*}}, col:14> col:5 used OK_1 'int ({{.*}})'
// CHECK-NEXT: |-FunctionDecl [[ADDR_28:0x[a-z0-9]*]] <line:22:1, col:14> col:5 used OK_3 'int ({{.*}})'
// CHECK-NEXT: `-FunctionDecl [[ADDR_30:0x[a-z0-9]*]] <line:24:1, line:27:1> line:24:5 test 'int ({{.*}})'
// CHECK-NEXT:   `-CompoundStmt [[ADDR_31:0x[a-z0-9]*]] <col:16, line:27:1>
// CHECK-NEXT:     `-ReturnStmt [[ADDR_32:0x[a-z0-9]*]] <line:26:3, col:35>
// C:                `-BinaryOperator [[ADDR_33:0x[a-z0-9]*]] <col:10, col:35> 'int' '+'
// C-NEXT:             |-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <col:10, col:26> 'int' '+'
// CXX:              `-BinaryOperator [[ADDR_33:0x[a-z0-9]*]] <col:10, col:35> '<dependent type>' contains-errors '+'
// CXX-NEXT:           |-BinaryOperator [[ADDR_34:0x[a-z0-9]*]] <col:10, col:26> '<dependent type>' contains-errors '+'
// CHECK-NEXT:         | |-CallExpr [[ADDR_35:0x[a-z0-9]*]] <col:10, col:15> 'int'
// CHECK-NEXT:         | | `-ImplicitCastExpr [[ADDR_36:0x[a-z0-9]*]] <col:10> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:         | |   `-DeclRefExpr [[ADDR_37:0x[a-z0-9]*]] <col:10> 'int ({{.*}})' {{.*}}Function [[ADDR_0]] 'OK_1' 'int ({{.*}})'
// C:                  | `-CallExpr [[ADDR_38:0x[a-z0-9]*]] <col:19, col:26> 'int'
// C-NEXT:             |   `-ImplicitCastExpr [[ADDR_39:0x[a-z0-9]*]] <col:19> 'int (*)({{.*}})' <FunctionToPointerDecay>
// C-NEXT:             |     `-DeclRefExpr [[ADDR_40:0x[a-z0-9]*]] <col:19> 'int ({{.*}})' {{.*}}Function {{.*}} 'not_OK' 'int ({{.*}})'
// CXX:                | `-RecoveryExpr {{.*}} <col:19, col:26> '<dependent type>' contains-errors lvalue
// CXX-NEXT:           |   `-UnresolvedLookupExpr {{.*}} <col:19> '<overloaded function type>' lvalue (ADL) = 'not_OK' empty
// CHECK-NEXT:         `-CallExpr [[ADDR_41:0x[a-z0-9]*]] <col:30, col:35> 'int'
// CHECK-NEXT:           `-ImplicitCastExpr [[ADDR_42:0x[a-z0-9]*]] <col:30> 'int (*)({{.*}})' <FunctionToPointerDecay>
// CHECK-NEXT:             `-DeclRefExpr [[ADDR_43:0x[a-z0-9]*]] <col:30> 'int ({{.*}})' {{.*}}Function [[ADDR_28]] 'OK_3' 'int ({{.*}})'
