// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -mdisable-tail-calls -o - | FileCheck %s -check-prefix=DISABLE
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s -check-prefix=ENABLE

// DISABLE: define{{.*}} i32 @f1() [[ATTRTRUE:#[0-9]+]] {
// DISABLE: define{{.*}} i32 @f2() [[ATTRTRUE]] {
// ENABLE: define{{.*}} i32 @f1() [[ATTRFALSE:#[0-9]+]] {
// ENABLE: define{{.*}} i32 @f2() [[ATTRTRUE:#[0-9]+]] {

int f1() {
  return 0;
}

int f2() __attribute__((disable_tail_calls)) {
  return 0;
}

// DISABLE: attributes [[ATTRTRUE]] = { {{.*}}"disable-tail-calls"="true"{{.*}} }
// ENABLE: attributes [[ATTRFALSE]] = { {{.*}}"disable-tail-calls"="false"{{.*}} }
// ENABLE: attributes [[ATTRTRUE]] = { {{.*}}"disable-tail-calls"="true"{{.*}} }
