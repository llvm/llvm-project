// RUN: %clang_cc1 -ast-dump %s  2>&1 | FileCheck %s

struct Sock {};
void leakNewFn() { new struct Sock; }
// CHECK: CXXNewExpr {{.*}} <col:20, col:31> 'struct Sock *'
