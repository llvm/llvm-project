// RUN: not %clang_cc1 -ast-dump=json %s 2>&1 | FileCheck %s

// Ensure that dumping this does not crash when emitting the mangled name.
// See GH137320 for details.
// Note, this file does not compile and so we also check the error.

@interface SomeClass (SomeExtension)
+ (void)someMethod;
@end

// CHECK: error: cannot find interface declaration for 'SomeClass'

// CHECK: "name": "someMethod"
// CHECK-NEXT: "mangledName": "+[ someMethod]",
