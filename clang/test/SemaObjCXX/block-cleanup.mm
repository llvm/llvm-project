// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11.0 -std=gnu++11 -o /dev/null -x objective-c++ -fblocks -ast-dump %s 2>&1 | FileCheck %s

// CHECK:      -FunctionDecl {{.*}} test 'id ()'
// CHECK-NEXT:   -CompoundStmt
// CHECK-NEXT:     -ReturnStmt
// CHECK-NEXT:       -ExprWithCleanups
// CHECK-NEXT:         -cleanup Block
// CHECK-NEXT:         -cleanup Block

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

id test() {
  return @{@"a": [](){}, @"b": [](){}};
}
