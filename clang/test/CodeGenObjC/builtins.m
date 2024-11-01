// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

id objc_msgSend(id, SEL, ...);

void test0(id receiver, SEL sel, const char *str) {
  short s = ((short (*)(id, SEL, const char*)) objc_msgSend)(receiver, sel, str);
}
// CHECK-LABEL: define{{.*}} void @test0(
// CHECK:   call signext i16 @objc_msgSend(
