// RUN: clang-format -assume-filename=foo.m -dump-config | FileCheck %s

// RUN: clang-format -dump-config - < %s | FileCheck %s

// CHECK: Language: ObjC

@interface Foo
@end
