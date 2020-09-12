// RUN: %clang_cc1 -fpass-by-value-is-noalias -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns -fobjc-runtime-has-weak -fobjc-arc -fobjc-dispatch-method=mixed %s -o - 2>&1 | FileCheck --check-prefix=WITH_NOALIAS %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns -fobjc-runtime-has-weak -fobjc-arc -fobjc-dispatch-method=mixed %s -o - 2>&1 | FileCheck --check-prefix=NO_NOALIAS %s

@interface Bar
@property char value;
@end

// A struct large enough so it is not passed in registers on ARM64, but with a
// weak reference, so noalias should not be added even with
// -fpass-by-value-is-noalias.
struct Foo {
  int a;
  int b;
  int c;
  int d;
  int e;
  Bar *__weak f;
};

// WITH_NOALIAS: define void @take(%struct.Foo* %arg)
// NO_NOALIAS: define void @take(%struct.Foo* %arg)
void take(struct Foo arg) {}
