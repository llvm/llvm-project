// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

void foo1(void*);
void foo2(void* const);


void bar() {
  // CHECK:  FunctionDecl {{.*}} <line:{{.*}}, line:{{.*}}> line:{{.*}} bar 'void ()'

  foo1(0);
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'void *' <NullToPointer>

  foo2(0);
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'void *' <NullToPointer>
}
