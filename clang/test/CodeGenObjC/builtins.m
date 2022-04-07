// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

void test0(id receiver, SEL sel, const char *str) {
  short s = ((short (*)(id, SEL, const char*)) objc_msgSend)(receiver, sel, str);
}
// CHECK-LABEL: define{{.*}} void @test0(
// CHECK:   call signext i16 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i16 (i8*, i8*, i8*)*)(
