@class Foo;
@interface Bar
@property (retain) __attribute__((iboutletcollection(Foo))) Foo *prop;
@end

@interface I
-(id)prop __attribute__((annotate("anno")));
-(void)setProp:(id)p __attribute__((annotate("anno")));
@property (assign) id prop __attribute__((annotate("anno")));
@end

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK:      <attribute>: attribute(iboutletcollection)= [IBOutletCollection=ObjCInterface]

// CHECK: <attribute>: attribute(annotate)=anno
// CHECK: <getter>: kind: objc-instance-method | name: prop | {{.*}} <attribute>: attribute(annotate)=anno
// CHECK: <setter>: kind: objc-instance-method | name: setProp: | {{.*}} <attribute>: attribute(annotate)=anno
