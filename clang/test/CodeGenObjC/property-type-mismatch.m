// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s
// rdar://8966864

@interface Foo
-(float)myfo;
-(void)setMyfo: (int)p;
@end

void bar(Foo *x) {
  x.myfo++;
}

// CHECK: [[C1:%.*]] = call float bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK: [[I:%.*]] = fadd float [[C1]], 1.000000e+00
// CHECK: [[CONV:%.*]] = fptosi float [[I]] to i32
// CHECK: [[T3:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_2"
// CHECK:  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
