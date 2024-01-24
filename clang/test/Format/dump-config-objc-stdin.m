// RUN: clang-format -dump-config - < %s | FileCheck %s

// CHECK: Language: ObjC
@interface Foo
@end
