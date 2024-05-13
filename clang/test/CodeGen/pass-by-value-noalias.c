// RUN: %clang_cc1 -fpass-by-value-is-noalias -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns %s -o - 2>&1 | FileCheck --check-prefix=WITH_NOALIAS %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -emit-llvm -disable-llvm-optzns %s -o - 2>&1 | FileCheck --check-prefix=NO_NOALIAS %s

// A struct large enough so it is not passed in registers on ARM64.
struct Foo {
  int a;
  int b;
  int c;
  int d;
  int e;
  int f;
};

// WITH_NOALIAS: define{{.*}} void @take(ptr noalias noundef %arg)
// NO_NOALIAS: define{{.*}} void @take(ptr noundef %arg)
void take(struct Foo arg) {}
