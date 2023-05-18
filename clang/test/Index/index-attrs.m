@class Foo;
@interface Bar
@property (retain) __attribute__((iboutletcollection(Foo))) Foo *prop;
@end

@interface I
-(id)prop __attribute__((annotate("anno")));
-(void)setProp:(id)p __attribute__((annotate("anno")));
@property (assign) id prop __attribute__((annotate("anno")));
@end

__attribute__((objc_protocol_requires_explicit_implementation))
@protocol P
@end

typedef id __attribute__((objc_independent_class)) T2;
id __attribute__((objc_precise_lifetime)) x;
struct __attribute__((objc_boxable)) S {
  int x;
};

__attribute__((objc_exception))
__attribute__((objc_root_class))
__attribute__((objc_subclassing_restricted))
__attribute__((objc_runtime_visible))
@interface J
-(id)a __attribute__((ns_returns_retained));
-(id)b __attribute__((ns_returns_not_retained));
-(id)c __attribute__((ns_returns_autoreleased));
-(id)d __attribute__((ns_consumes_self));
-(id)e __attribute__((objc_requires_super));
-(int *)f __attribute__((objc_returns_inner_pointer));
-(id)init __attribute__((objc_designated_initializer));
@end

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK:      <attribute>: attribute(iboutletcollection)= [IBOutletCollection=ObjCInterface]

// CHECK: <attribute>: attribute(annotate)=anno
// CHECK: <getter>: kind: objc-instance-method | name: prop | {{.*}} <attribute>: attribute(annotate)=anno
// CHECK: <setter>: kind: objc-instance-method | name: setProp: | {{.*}} <attribute>: attribute(annotate)=anno
// CHECK: <attribute>: attribute(objc_protocol_requires_explicit_implementation)=
// CHECK: <attribute>: attribute(objc_independent_class)=
// CHECK: <attribute>: attribute(objc_precise_lifetime)=
// CHECK: <attribute>: attribute(objc_boxable)=
// CHECK: <attribute>: attribute(objc_exception)=
// CHECK: <attribute>: attribute(objc_root_class)=
// CHECK: <attribute>: attribute(objc_subclassing_restricted)=
// CHECK: <attribute>: attribute(objc_runtime_visible)=
// CHECK: <attribute>: attribute(ns_returns_retained)=
// CHECK: <attribute>: attribute(ns_returns_not_retained)=
// CHECK: <attribute>: attribute(ns_returns_autoreleased)=
// CHECK: <attribute>: attribute(ns_consumes_self)=
// CHECK: <attribute>: attribute(objc_requires_super)=
// CHECK: <attribute>: attribute(objc_returns_inner_pointer)=
// CHECK: <attribute>: attribute(objc_designated_initializer)=
