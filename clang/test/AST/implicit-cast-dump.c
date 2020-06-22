// Test without serialization:
// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

void foo1(void*);
void foo2(void* const);


void bar() {
  // CHECK:  FunctionDecl {{.*}} <line:{{.*}}, line:{{.*}}> line:{{.*}} bar 'void ()'

  foo1(0);
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'void *' <NullToPointer>

  foo2(0);
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'void *' <NullToPointer>
}
