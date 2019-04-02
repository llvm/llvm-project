// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify %s
// rdar://9340606

@interface Foo {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(strong) id x;
@property(strong) id y;
@property(strong) id z;
@end

@interface Bar {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(retain) id x;
@property(retain) id y;
@property(retain) id z;
@end

@interface Bas {
@public
    id __unsafe_unretained x;
    id __weak y;
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(copy) id x;
@property(copy) id y;
@property(copy) id z;
@end

// Errors should start about here :-)

@interface Bat 
@property(strong) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(strong) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(strong) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bau
@property(retain) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(retain) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(retain) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bav 
@property(copy) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(copy) __weak id y; // expected-error {{strong property 'y' may not also be declared __weak}}
@property(copy) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bingo 
@property(assign) __unsafe_unretained id x;
@property(assign) __weak id y; // expected-error {{unsafe_unretained property 'y' may not also be declared __weak}}
@property(assign) __autoreleasing id z; // expected-error {{unsafe_unretained property 'z' may not also be declared __autoreleasing}}
@end

@interface Batman 
@property(unsafe_unretained) __unsafe_unretained id x;
@property(unsafe_unretained) __weak id y; // expected-error {{unsafe_unretained property 'y' may not also be declared __weak}}
@property(unsafe_unretained) __autoreleasing id z; // expected-error {{unsafe_unretained property 'z' may not also be declared __autoreleasing}}
@end

// rdar://9396329
@interface Super
@property (readonly, retain) id foo;
@property (readonly, weak) id fee;
@property (readonly, strong) id frr;
@end

@interface Bugg : Super
@property (readwrite) id foo;
@property (readwrite) id fee;
@property (readwrite) id frr;
@end

// rdar://20152386
// rdar://20383235

@interface NSObject @end

#pragma clang assume_nonnull begin
@interface I: NSObject
@property(nonatomic, weak) id delegate; // Do not warn, nullable is inferred. 
@property(nonatomic, weak, readonly) id ROdelegate; // Do not warn, nullable is inferred.
@property(nonatomic, weak, nonnull) id NonNulldelete; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@property(nonatomic, weak, nullable) id Nullabledelete; // do not warn

// strong cases.
@property(nonatomic, strong) id stdelegate; // Do not warn
@property(nonatomic, readonly) id stROdelegate; // Do not warn
@property(nonatomic, strong, nonnull) id stNonNulldelete; // Do not warn
@property(nonatomic, nullable) id stNullabledelete; // do not warn
@end
#pragma clang assume_nonnull end

@interface J: NSObject
@property(nonatomic, weak) id ddd;   // Do not warn, nullable is inferred.
@property(nonatomic, weak, nonnull) id delegate; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@property(nonatomic, weak, nonnull, readonly) id ROdelegate; // expected-error {{property attributes 'nonnull' and 'weak' are mutually exclusive}}
@end

// rdar://problem/23931441
@protocol P
@property(readonly, retain) id prop;
@end

__attribute__((objc_root_class))
@interface I2<P>
@end

@interface I2()
@property (readwrite) id prop;
@end

@implementation I2
@synthesize prop;
@end

// rdar://31579994
// Verify that the all of the property declarations in inherited protocols are
// compatible when synthesing a property from a protocol.

@protocol CopyVsAssign1
@property (copy, nonatomic,  readonly) id prop; // expected-error {{property with attribute 'copy' was selected for synthesis}}
@end
@protocol CopyVsAssign2
@property (assign, nonatomic, readonly) id prop; // expected-note {{it could also be property without attribute 'copy' declared here}}
@end

@interface CopyVsAssign: Foo <CopyVsAssign1, CopyVsAssign2>
@end
@implementation CopyVsAssign
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol RetainVsNonRetain1
@property (readonly) id prop; // expected-error {{property without attribute 'retain (or strong)' was selected for synthesis}}
@end
@protocol RetainVsNonRetain2
@property (retain, readonly) id prop; // expected-note {{it could also be property with attribute 'retain (or strong)' declared here}}
@end

@interface RetainVsNonRetain: Foo <RetainVsNonRetain1, RetainVsNonRetain2>
@end
@implementation RetainVsNonRetain
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol AtomicVsNonatomic1
@property (copy, nonatomic, readonly) id prop; // expected-error {{property without attribute 'atomic' was selected for synthesis}}
@end
@protocol AtomicVsNonatomic2
@property (copy, atomic, readonly) id prop; // expected-note {{it could also be property with attribute 'atomic' declared here}}
@end

@interface AtomicVsNonAtomic: Foo <AtomicVsNonatomic1, AtomicVsNonatomic2>
@end
@implementation AtomicVsNonAtomic
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol Getter1
@property (copy, readonly) id prop; // expected-error {{property with getter 'prop' was selected for synthesis}}
@end
@protocol Getter2
@property (copy, getter=x, readonly) id prop; // expected-note {{it could also be property with getter 'x' declared here}}
@end

@interface GetterVsGetter: Foo <Getter1, Getter2>
@end
@implementation GetterVsGetter
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol Setter1
@property (copy, readonly) id prop;
@end
@protocol Setter2
@property (copy, setter=setp:, readwrite) id prop; // expected-error {{property with setter 'setp:' was selected for synthesis}}
@end
@protocol Setter3
@property (copy, readwrite) id prop; // expected-note {{it could also be property with setter 'setProp:' declared here}}
@end

@interface SetterVsSetter: Foo <Setter1, Setter2, Setter3>
@end
@implementation SetterVsSetter
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol TypeVsAttribute1
@property (assign, atomic, readonly) int prop; // expected-error {{property of type 'int' was selected for synthesis}}
@end
@protocol TypeVsAttribute2
@property (assign, atomic, readonly) id prop; // expected-note {{it could also be property of type 'id' declared here}}
@end
@protocol TypeVsAttribute3
@property (copy, readonly) id prop; // expected-note {{it could also be property with attribute 'copy' declared here}}
@end

@interface TypeVsAttribute: Foo <TypeVsAttribute1, TypeVsAttribute2, TypeVsAttribute3>
@end
@implementation TypeVsAttribute
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol TypeVsSetter1
@property (assign, nonatomic, readonly) int prop; // expected-note {{it could also be property of type 'int' declared here}}
@end
@protocol TypeVsSetter2
@property (assign, nonatomic, readonly) id prop; // ok
@end
@protocol TypeVsSetter3
@property (assign, nonatomic, readwrite) id prop; // expected-error {{property of type 'id' was selected for synthesis}}
@end

@interface TypeVsSetter: Foo <TypeVsSetter1, TypeVsSetter2, TypeVsSetter3>
@end
@implementation TypeVsSetter
@synthesize prop; // expected-note {{property synthesized here}}
@end

@protocol AutoStrongProp

@property (nonatomic, readonly) NSObject *prop;

@end

@protocol AutoStrongProp_Internal <AutoStrongProp>

// This property gets the 'strong' attribute automatically.
@property (nonatomic, readwrite) NSObject *prop;

@end

@interface SynthesizeWithImplicitStrongNoError : NSObject <AutoStrongProp>
@end

@interface SynthesizeWithImplicitStrongNoError () <AutoStrongProp_Internal>

@end

@implementation SynthesizeWithImplicitStrongNoError

// no error, 'strong' is implicit in the 'readwrite' property.
@synthesize prop = _prop;

@end

// rdar://39024725
// Allow strong readwrite property and a readonly one.
@protocol StrongCollision

@property(strong) NSObject *p;
@property(copy) NSObject *p2;

// expected-error@+1 {{property with attribute 'retain (or strong)' was selected for synthesis}}
@property(strong, readwrite) NSObject *collision;

@end

@protocol ReadonlyCollision

@property(readonly) NSObject *p;
@property(readonly) NSObject *p2;

// expected-note@+1 {{it could also be property without attribute 'retain (or strong)' declared here}}
@property(readonly, weak) NSObject *collision;

@end

@interface StrongReadonlyCollision : NSObject <StrongCollision, ReadonlyCollision>
@end

@implementation StrongReadonlyCollision

// no error
@synthesize p = _p;
@synthesize p2 = _p2;

@synthesize collision = _collision; // expected-note {{property synthesized here}}

@end

// This used to crash because we'd temporarly store the weak attribute on the
// declaration specifier, then deallocate it when clearing the declarator.
id i1, __weak i2, i3;
