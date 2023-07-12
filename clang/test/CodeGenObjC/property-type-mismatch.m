// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s

@interface Foo
-(float)myfo;
-(void)setMyfo: (int)p;
@end

void bar(Foo *x) {
  x.myfo++;
}

// CHECK: [[C1:%.*]] = call float @objc_msgSend
// CHECK: [[I:%.*]] = fadd float [[C1]], 1.000000e+00
// CHECK: [[CONV:%.*]] = fptosi float [[I]] to i32
// CHECK: [[T3:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.2
// CHECK:  call void @objc_msgSend
