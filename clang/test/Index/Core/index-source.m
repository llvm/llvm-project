// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

@interface Base
// CHECK: [[@LINE-1]]:12 | objc-class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Decl | rel: 0
-(void)meth;
// CHECK: [[@LINE-1]]:1 | objc-instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Decl/Dyn/RelChild | rel: 1
// CHECK-NEXT: RelChild | Base | c:objc(cs)Base
@end
