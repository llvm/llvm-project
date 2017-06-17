// RUN: %clang -target x86_64-unknown-windows-msvc -fobjc-runtime=ios -Wno-objc-root-class -S -o - -emit-llvm %s | FileCheck %s
// RUN: %clang -target x86_64-apple-ios -fobjc-runtime=ios -Wno-objc-root-class -S -o - -emit-llvm %s | FileCheck %s

struct S {
  float f, g;
};

@interface I
@property struct S s;
@end

@implementation I
@end

// CHECK: declare {{.*}}void @objc_copyStruct(i8*, i8*, i64, i1, i1)

