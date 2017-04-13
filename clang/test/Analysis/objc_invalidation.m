// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.osx.cocoa.InstanceVariableInvalidation -DRUN_IVAR_INVALIDATION -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.osx.cocoa.MissingInvalidationMethod -DRUN_MISSING_INVALIDATION_METHOD -verify %s
extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));

#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))

@protocol NSObject
@end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
-(id)description;
@end
@class NSString;

extern void NSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));

@protocol Invalidation1 <NSObject> 
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end 

@protocol Invalidation2 <NSObject> 
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end 

@protocol Invalidation3 <NSObject>
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
- (void) invalidate2 __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@protocol Invalidation3;
@protocol Invalidation2;

@interface Invalidation2Class <Invalidation2>
@end

@interface Invalidation1Class <Invalidation1>
@end

@interface ClassWithInvalidationMethodInCategory <NSObject>
@end

@interface ClassWithInvalidationMethodInCategory ()
- (void) invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@interface SomeInvalidationImplementingObject: NSObject <Invalidation3, Invalidation2> {
  SomeInvalidationImplementingObject *ObjA; // invalidation in the parent
}
@end

@implementation SomeInvalidationImplementingObject
- (void)invalidate{
  ObjA = 0;
}
- (void)invalidate2 {
  [self invalidate];
}
@end

@interface SomeSubclassInvalidatableObject : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Ivar1; // regular ivar
  SomeInvalidationImplementingObject *Ivar2; // regular ivar, sending invalidate message
  SomeInvalidationImplementingObject *_Ivar3; // no property, call -description
  SomeInvalidationImplementingObject *_Ivar4; // no property, provide as argument to NSLog()

  SomeInvalidationImplementingObject *_Prop1; // partially implemented property, set to 0 with dot syntax
  SomeInvalidationImplementingObject *_Prop2; // fully implemented prop, set to 0 with dot syntax
  SomeInvalidationImplementingObject *_propIvar; // property with custom named ivar, set to 0 via setter
  Invalidation1Class *MultipleProtocols; // regular ivar belonging to a different class
  Invalidation2Class *MultInheritance; // regular ivar belonging to a different class
  SomeInvalidationImplementingObject *_Prop3; // property, invalidate via sending a message to a getter method
  SomeInvalidationImplementingObject *_Prop4; // property with @synthesize, invalidate via property
  SomeInvalidationImplementingObject *_Prop5; // property with @synthesize, invalidate via getter method
  SomeInvalidationImplementingObject *_Prop8;
  
  // Ivars invalidated by the partial invalidator. 
  SomeInvalidationImplementingObject *Ivar9;
  SomeInvalidationImplementingObject *_Prop10;
  SomeInvalidationImplementingObject *Ivar11;

  // No warnings on these as they are not invalidatable.
  NSObject *NIvar1;
  NSObject *NObj2;
  NSObject *_NProp1;
  NSObject *_NpropIvar;
}

@property (assign) SomeInvalidationImplementingObject* Prop0;
@property (nonatomic, assign) SomeInvalidationImplementingObject* Prop1;
@property (assign) SomeInvalidationImplementingObject* Prop2;
@property (assign) SomeInvalidationImplementingObject* Prop3;
@property (assign) SomeInvalidationImplementingObject *Prop5;
@property (assign) SomeInvalidationImplementingObject *Prop4;

@property (assign) SomeInvalidationImplementingObject* Prop6; // automatically synthesized prop
@property (assign) SomeInvalidationImplementingObject* Prop7; // automatically synthesized prop
@property (assign) SomeInvalidationImplementingObject *SynthIvarProp;

@property (assign) NSObject* NProp0;
@property (nonatomic, assign) NSObject* NProp1;
@property (assign) NSObject* NProp2;

-(void)setProp1: (SomeInvalidationImplementingObject*) InO;
-(void)setNProp1: (NSObject*) InO;

-(void)invalidate;

// Partial invalidators invalidate only some ivars. They are guaranteed to be 
// called before the invalidation methods.
-(void)partialInvalidator1 __attribute__((annotate("objc_instance_variable_invalidator_partial")));
-(void)partialInvalidator2 __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end

@interface SomeSubclassInvalidatableObject()
@property (assign) SomeInvalidationImplementingObject* Prop8;
@property (assign) SomeInvalidationImplementingObject* Prop10;
@end

@implementation SomeSubclassInvalidatableObject{
  @private
  SomeInvalidationImplementingObject *Ivar5;
  ClassWithInvalidationMethodInCategory *Ivar13;
}

@synthesize Prop7 = _propIvar;
@synthesize Prop3 = _Prop3;
@synthesize Prop5 = _Prop5;
@synthesize Prop4 = _Prop4;
@synthesize Prop8 = _Prop8;
@synthesize Prop10 = _Prop10;


- (void) setProp1: (SomeInvalidationImplementingObject*) InObj {
  _Prop1 = InObj;
}

- (void) setProp2: (SomeInvalidationImplementingObject*) InObj {
  _Prop2 = InObj;
}
- (SomeInvalidationImplementingObject*) Prop2 {
  return _Prop2;
}

@synthesize NProp2 = _NpropIvar;

- (void) setNProp1: (NSObject*) InObj {
  _NProp1 = InObj;
}

- (void) invalidate {
   [Ivar2 invalidate];
   self.Prop0 = 0;
   self.Prop1 = 0;
   [self setProp2:0];
   [self setProp3:0];
   [[self Prop5] invalidate2];
   [self.Prop4 invalidate];
   [self.Prop8 invalidate];
   self.Prop6 = 0;
   [[self Prop7] invalidate];

   [_Ivar3 description]; 
   NSLog(@"%@", _Ivar4);
   [super invalidate];
}
#if RUN_IVAR_INVALIDATION
// expected-warning@-2 {{Instance variable Ivar1 needs to be invalidated}}
// expected-warning@-3 {{Instance variable MultipleProtocols needs to be invalidated}}
// expected-warning@-4 {{Instance variable MultInheritance needs to be invalidated}}
// expected-warning@-5 {{Property SynthIvarProp needs to be invalidated or set to nil}}
// expected-warning@-6 {{Instance variable _Ivar3 needs to be invalidated}}
// expected-warning@-7 {{Instance variable _Ivar4 needs to be invalidated}}
// expected-warning@-8 {{Instance variable Ivar5 needs to be invalidated or set to nil}}
// expected-warning@-9 {{Instance variable Ivar13 needs to be invalidated or set to nil}}
#endif

-(void)partialInvalidator1 {
  [Ivar9 invalidate];
  [_Prop10 invalidate];
}

-(void)partialInvalidator2 {
  [Ivar11 invalidate];
}

@end

// Example, where the same property is inherited through 
// the parent and directly through a protocol. If a property backing ivar is 
// synthesized in the parent, let the parent invalidate it.

@protocol IDEBuildable <NSObject>
@property (readonly, strong) id <Invalidation2> ObjB;
@end

@interface Parent : NSObject <IDEBuildable, Invalidation2> {
  Invalidation2Class *_ObjB; // Invalidation of ObjB happens in the parent.
}
@end

@interface Child: Parent <Invalidation2, IDEBuildable> 
@end

@implementation Parent{
  @private
  Invalidation2Class *Ivar10;
  Invalidation2Class *Ivar11;
  Invalidation2Class *Ivar12;
}

@synthesize ObjB = _ObjB;
- (void)invalidate{
  _ObjB = ((void*)0);
  
  assert(Ivar10 == 0);

  if (__builtin_expect(!(Ivar11 == ((void*)0)), 0))
    assert(0);

  assert(0 == Ivar12);

}
@end

@implementation Child
- (void)invalidate{ 
  // no-warning
} 
@end

@protocol Invalidation <NSObject>
- (void)invalidate __attribute__((annotate("objc_instance_variable_invalidator")));
@end

@interface Foo : NSObject <Invalidation>
@end

@class FooBar;
@protocol FooBar_Protocol <NSObject>
@end

@interface MissingInvalidationMethod : Foo <FooBar_Protocol>
@property (assign) MissingInvalidationMethod *foobar15_warn;
#if RUN_IVAR_INVALIDATION
// expected-warning@-2 {{Property foobar15_warn needs to be invalidated; no invalidation method is defined in the @implementation for MissingInvalidationMethod}}
#endif
@end
@implementation MissingInvalidationMethod
@end

@interface MissingInvalidationMethod2 : Foo <FooBar_Protocol> {
  Foo *Ivar1;
#if RUN_IVAR_INVALIDATION
// expected-warning@-2 {{Instance variable Ivar1 needs to be invalidated; no invalidation method is defined in the @implementation for MissingInvalidationMethod2}}
#endif
}
@end
@implementation MissingInvalidationMethod2
@end

@interface MissingInvalidationMethodDecl : NSObject {
  Foo *Ivar1;
#if RUN_MISSING_INVALIDATION_METHOD
// expected-warning@-2 {{Instance variable Ivar1 needs to be invalidated; no invalidation method is declared for MissingInvalidationMethodDecl}}
#endif
}
@end
@implementation MissingInvalidationMethodDecl
@end

@interface MissingInvalidationMethodDecl2 : NSObject {
@private
    Foo *_foo1;
#if RUN_MISSING_INVALIDATION_METHOD
// expected-warning@-2 {{Instance variable _foo1 needs to be invalidated; no invalidation method is declared for MissingInvalidationMethodDecl2}}
#endif
}
@property (strong) Foo *bar1; 
@end
@implementation MissingInvalidationMethodDecl2
@end

@interface InvalidatedInPartial : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Ivar1; 
  SomeInvalidationImplementingObject *Ivar2; 
}
-(void)partialInvalidator __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end
@implementation InvalidatedInPartial
-(void)partialInvalidator {
  [Ivar1 invalidate];
  Ivar2 = 0;
}
@end

@interface NotInvalidatedInPartial : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Ivar1; 
}
-(void)partialInvalidator __attribute__((annotate("objc_instance_variable_invalidator_partial")));
-(void)partialInvalidatorCallsPartial __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end
@implementation NotInvalidatedInPartial
-(void)partialInvalidator {
}
-(void)partialInvalidatorCallsPartial {
  [self partialInvalidator];
}

-(void)invalidate {
} 
#if RUN_IVAR_INVALIDATION
// expected-warning@-2 {{Instance variable Ivar1 needs to be invalidated or set to nil}}
#endif
@end

@interface SomeNotInvalidatedInPartial : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Ivar1;
  SomeInvalidationImplementingObject *Ivar2;
#if RUN_IVAR_INVALIDATION
  // expected-warning@-2 {{Instance variable Ivar2 needs to be invalidated or set to nil}}
#endif
}
-(void)partialInvalidator __attribute__((annotate("objc_instance_variable_invalidator_partial")));
-(void)partialInvalidatorCallsPartial __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end
@implementation SomeNotInvalidatedInPartial {
  SomeInvalidationImplementingObject *Ivar3;
#if RUN_IVAR_INVALIDATION
  // expected-warning@-2 {{Instance variable Ivar3 needs to be invalidated or set to nil}}
#endif
}
-(void)partialInvalidator {
  Ivar1 = 0;
}
-(void)partialInvalidatorCallsPartial {
  [self partialInvalidator];
}
@end

@interface OnlyPartialDeclsBase : NSObject
-(void)partialInvalidator __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end
@implementation OnlyPartialDeclsBase
-(void)partialInvalidator {}
@end

@interface OnlyPartialDecls : OnlyPartialDeclsBase {
  SomeInvalidationImplementingObject *Ivar1;
#if RUN_IVAR_INVALIDATION
  // expected-warning@-2 {{Instance variable Ivar1 needs to be invalidated; no invalidation method is defined in the @implementation for OnlyPartialDecls}}
#endif
}
@end
@implementation OnlyPartialDecls
@end

// False negative.
@interface PartialCallsFull : SomeInvalidationImplementingObject {
  SomeInvalidationImplementingObject *Ivar1;
}
-(void)partialInvalidator __attribute__((annotate("objc_instance_variable_invalidator_partial")));
@end
@implementation PartialCallsFull
-(void)partialInvalidator {
 [self invalidate];
} // TODO: It would be nice to check that the full invalidation method actually invalidates the ivar. 
@end

