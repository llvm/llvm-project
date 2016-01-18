// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -mdisable-tail-calls -o - | FileCheck %s -check-prefix=CHECK -check-prefix=DISABLE
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK -check-prefix=ENABLE

// CHECK: define i32 @f1() [[ATTR:#[0-9]+]] {

int f1() {
  return 0;
}

// DISABLE: attributes [[ATTR]] = { {{.*}} "disable-tail-calls"="true" {{.*}} }
// ENABLE: attributes [[ATTR]] = { {{.*}} "disable-tail-calls"="false" {{.*}} }
