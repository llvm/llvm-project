// RUN: cat %s | \
// RUN:   clang-repl -Xcc -fno-color-diagnostics -Xcc -Xclang -Xcc -ast-dump \
// RUN:            -Xcc -Xclang -Xcc -ast-dump-filter -Xcc -Xclang -Xcc Test 2>&1| \
// RUN:         FileCheck %s

int TestVar = 12;
// CHECK: Dumping TestVar:
// CHECK-NEXT: VarDecl [[var_ptr:0x[0-9a-f]+]] <{{.*}} TestVar 'int' cinit
// CHECK-NEXT:   IntegerLiteral {{.*}} 'int' 12

void TestFunc() { ++TestVar; }
// CHECK: Dumping TestFunc:
// CHECK-NEXT: FunctionDecl {{.*}} TestFunc 'void ()'
// CHECK-NEXT:   CompoundStmt{{.*}}
// CHECK-NEXT:     UnaryOperator{{.*}} 'int' lvalue prefix '++'
// CHECK-NEXT:       DeclRefExpr{{.*}} 'int' lvalue Var [[var_ptr]] 'TestVar' 'int'

quit
