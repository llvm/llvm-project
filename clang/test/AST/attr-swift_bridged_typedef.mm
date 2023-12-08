// RUN: %clang_cc1 -fsyntax-only %s -ast-dump | FileCheck %s

@interface NSString
@end

using NSStringAlias __attribute__((__swift_bridged_typedef__)) = NSString *;
// CHECK: TypeAliasDecl {{.*}} NSStringAlias 'NSString *'
// CHECK: SwiftBridgedTypedefAttr
