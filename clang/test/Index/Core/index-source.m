// RUN: c-index-test core -print-source-symbols -- %s -target x86_64-apple-macosx10.7 | FileCheck %s

@interface Base
// CHECK: [[@LINE-1]]:12 | objc-class/ObjC | Base | c:objc(cs)Base | _OBJC_CLASS_$_Base | Decl | rel: 0
-(void)meth;
// CHECK: [[@LINE-1]]:1 | objc-instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Decl,Dyn,RelChild | rel: 1
// CHECK-NEXT: RelChild | Base | c:objc(cs)Base
@end

void foo();
// CHECK: [[@LINE+1]]:6 | function/C | goo | c:@F@goo | _goo | Def | rel: 0
void goo(Base *b) {
  // CHECK: [[@LINE+1]]:3 | function/C | foo | c:@F@foo | _foo | Ref,Call | rel: 0
  foo();
  // CHECK: [[@LINE+2]]:6 | objc-instance-method/ObjC | meth | c:objc(cs)Base(im)meth | -[Base meth] | Ref,Call,Dyn,RelRec | rel: 1
  // CHECK-NEXT: RelRec | Base | c:objc(cs)Base
  [b meth];
}
