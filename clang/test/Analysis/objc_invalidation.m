// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.osx.cocoa.InstanceVariableInvalidation -fobjc-default-synthesize-properties -verify %s

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

@interface Invalidation2Class <Invalidation2>
@end

@interface Invalidation1Class <Invalidation1>
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

@end

@interface SomeSubclassInvalidatableObject()
@property (assign) SomeInvalidationImplementingObject* Prop8;
@end

@implementation SomeSubclassInvalidatableObject{
  @private
  SomeInvalidationImplementingObject *Ivar5;
}

@synthesize Prop7 = _propIvar;
@synthesize Prop3 = _Prop3;
@synthesize Prop5 = _Prop5;
@synthesize Prop4 = _Prop4;
@synthesize Prop8 = _Prop8;


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
// expected-warning@-1 {{Instance variable Ivar1 needs to be invalidated}}
 // expected-warning@-2 {{Instance variable MultipleProtocols needs to be invalidated}}
 // expected-warning@-3 {{Instance variable MultInheritance needs to be invalidated}}
 // expected-warning@-4 {{Property SynthIvarProp needs to be invalidated or set to nil}}
 // expected-warning@-5 {{Instance variable _Ivar3 needs to be invalidated}}
 // expected-warning@-6 {{Instance variable _Ivar4 needs to be invalidated}}
 // expected-warning@-7 {{Instance variable Ivar5 needs to be invalidated or set to nil}}
@end
