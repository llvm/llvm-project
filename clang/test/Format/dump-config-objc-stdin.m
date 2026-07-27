// RUN: clang-format -style=llvm -assume-filename=foo.m -dump-config | FileCheck %s

// RUN: clang-format -style=llvm -dump-config - < %s | FileCheck %s

// CHECK: Language: ObjC

@interface Foo
@end
