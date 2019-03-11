// RUN: %clang_cc1 -fsyntax-only -Wno-incomplete-implementation -verify -fblocks %s

#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))
#define NS_UNAVAILABLE __attribute__((unavailable))

void fnfoo(void) NS_DESIGNATED_INITIALIZER; // expected-error {{'objc_designated_initializer' attribute only applies to Objective-C methods}}

@protocol P1
-(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to init methods of interface or class extension declarations}}
@end

__attribute__((objc_root_class))
@interface I1
-(void)meth NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to init methods of interface or class extension declarations}}
-(id)init NS_DESIGNATED_INITIALIZER;
+(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to init methods of interface or class extension declarations}}
@end

@interface I1(cat)
-(id)init2 NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to init methods of interface or class extension declarations}}
@end

@interface I1()
-(id)init3 NS_DESIGNATED_INITIALIZER;
@end

@implementation I1
-(void)meth {}
-(id)init NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to init methods of interface or class extension declarations}}
+(id)init { return 0; }
-(id)init3 { return 0; }
-(id)init4 NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to init methods of interface or class extension declarations}} \
									 			   // expected-warning {{convenience initializer missing a 'self' call to another initializer}}
@end

__attribute__((objc_root_class))
@interface B1
-(id)initB1 NS_DESIGNATED_INITIALIZER; // expected-note 6 {{method marked as designated initializer of the class here}}
-(id)initB2;
@end

@interface B1()
-(id)initB3 NS_DESIGNATED_INITIALIZER; // expected-note 4 {{method marked as designated initializer of the class here}}
@end;

@implementation B1
-(id)initB1 { return 0; }
-(id)initB2 { return 0; } // expected-warning {{convenience initializer missing a 'self' call to another initializer}}
-(id)initB3 { return 0; }
@end

@interface S1 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER; // expected-note {{method marked as designated initializer of the class here}}
-(id)initS2 NS_DESIGNATED_INITIALIZER;
-(id)initS3 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
-(id)initB1;
@end

@interface S1()
-(id)initS4 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
@end

@implementation S1
-(id)initS1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initS2 {
  return [super initB1];
}
-(id)initS3 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [super initB2]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
-(id)initS4 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [self initB1]; // expected-warning {{designated initializer should only invoke a designated initializer on 'super'}}
}
-(id)initB1 {
  return [self initS1];
}
-(id)initB3 {
  return [self initS1];
}
@end

@interface S2 : B1
-(id)initB1;
@end

@interface SS2 : S2
-(id)initSS1 NS_DESIGNATED_INITIALIZER;
@end

@implementation SS2 // expected-warning {{method override for the designated initializer of the superclass '-initB1' not found}} \
                    // expected-warning {{method override for the designated initializer of the superclass '-initB3' not found}}
-(id)initSS1 {
  return [super initB1];
}
@end

@interface S3 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER; // expected-note {{method marked as designated initializer of the class here}}
@end

@interface SS3 : S3
-(id)initSS1 NS_DESIGNATED_INITIALIZER; // expected-note 2 {{method marked as designated initializer of the class here}}
@end

@implementation SS3 // expected-warning {{method override for the designated initializer of the superclass '-initS1' not found}}
-(id)initSS1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return [super initB1]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
@end

@interface S4 : B1
-(id)initB1;
-(id)initB3;
@end

@implementation S4
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initB3 {
  return [super initB3];
}
@end

@interface S5 : B1
-(void)meth;
@end

@implementation S5
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
-(id)initB3 {
  [self initB1]; // expected-warning {{designated initializer should only invoke a designated initializer on 'super'}}
  S5 *s;
  [s initB1];
  [self meth];
  void (^blk)(void) = ^{
    [self initB1]; // expected-warning {{designated initializer should only invoke a designated initializer on 'super'}}
  };
  return [super initB3];
}
-(void)meth {}
@end

@interface S6 : B1
-(id)initS1 NS_DESIGNATED_INITIALIZER;
-(id)initS2;
-(id)initS3;
-(id)initS4;
@end

@implementation S6 // expected-warning {{method override for the designated initializer of the superclass '-initB1' not found}} \
                   // expected-warning {{method override for the designated initializer of the superclass '-initB3' not found}}
-(id)initS1 {
  return [super initB1];
}
-(id)initS2 { // expected-warning {{convenience initializer missing a 'self' call to another initializer}}
  return [super initB1]; // expected-warning {{convenience initializer should not invoke an initializer on 'super'}}
}
-(id)initS3 {
  return [self initB1];
}
-(id)initS4 {
  return [self initS1];
}
-(id)initS5 {
  [super initB1]; // expected-warning {{convenience initializer should not invoke an initializer on 'super'}}
  void (^blk)(void) = ^{
    [super initB1]; // expected-warning {{convenience initializer should not invoke an initializer on 'super'}}
  };
  return [self initS1];
}
-(id)initS6 { // expected-warning {{convenience initializer missing a 'self' call to another initializer}}
  S6 *s;
  return [s initS1];
}
@end

@interface SS4 : S4
-(id)initB1;
@end

@implementation SS4
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
@end

@interface S7 : B1
-(id)initB1;
-(id)initB3;
-(id)initNewOne;
@end

@interface SS7 : S7
-(id)initB1;
@end

@implementation SS7
-(id)initB1 {
  return 0;
}
@end

__attribute__((objc_root_class))
@interface B2
-(id)init;
@end

@interface S8: B2
-(id)initS8 NS_DESIGNATED_INITIALIZER;
@end

@implementation S8
-(id)initS8
{
  return [super init];
}
@end

@interface S9 : B1
-(id)initB1;
-(id)initB3;
@end

@interface S9(secondInit)
-(id)initNewOne;
@end

@interface SS9 : S9
-(id)initB1;
@end

@implementation SS9
-(id)initB1 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return 0;
}
@end

// rdar://16261494
@class GEOPDAnalyticMetadata; // expected-note {{forward declaration of class here}}

@implementation GEOPDAnalyticMetadata (PlaceCardExtras) // expected-error {{cannot find interface declaration for 'GEOPDAnalyticMetadata'}}
- (instancetype)initInProcess
{
    return ((void*)0);
}
@end

// rdar://16305460
__attribute__((objc_root_class))
@interface MyObject
- (instancetype)initWithStuff:(id)stuff __attribute__((objc_designated_initializer));
- (instancetype)init NS_UNAVAILABLE;
@end

@implementation MyObject
- (instancetype)init
{
   return ((void*)0);
}
@end

// rdar://16323233
__attribute__((objc_root_class))
@interface B4 
-(id)initB4 NS_DESIGNATED_INITIALIZER; // expected-note 4 {{method marked as designated initializer of the class here}}
-(id)initNonDI;
@end

@interface rdar16323233 : B4
-(id)initS4 NS_DESIGNATED_INITIALIZER;
@end

@implementation rdar16323233
-(id)initS4 {
    static id sSharedObject = (void*)0;
    (void)^(void) {
        sSharedObject = [super initB4];
    };
    return 0;
}
-(id)initB4 {
   return [self initS4];
}
@end

@interface S1B4 : B4
@end
@implementation S1B4
-(id)initB4 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
   return [super initNonDI]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
@end

@interface S2B4 : B4
-(id)initB4;
@end
@implementation S2B4
-(id)initB4 { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
   return [super initNonDI]; // expected-warning {{designated initializer invoked a non-designated initializer}}
}
@end

@interface S3B4 : B4
@end
@implementation S3B4
-(id)initNew {
  return [super initB4];
}
-(id)initB4 {
   return [self initNew];
}
@end

@interface S4B4 : B4
-(id)initNew;
@end
@implementation S4B4
-(id)initNew {
  return [super initB4];
}
-(id)initB4 {
   return [self initNew];
}
@end

@interface S5B4 : B4
-(id)initB4;
@end
@implementation S5B4
-(id)initNew {
  return [super initB4];
}
-(id)initB4 {
   return [self initNew];
}
@end

@interface S6B4 : B4
-(id)initNew;
-(id)initB4;
@end
@implementation S6B4
-(id)initNew {
  return [super initB4];
}
-(id)initB4 {
   return [self initNew];
}
@end

__attribute__((objc_root_class))
@interface NSObject
-(instancetype) init NS_DESIGNATED_INITIALIZER; // expected-note {{method marked as designated initializer of the class here}}
@end

@interface Test3 : NSObject
@end

@implementation Test3
-(instancetype) initWithBasePath:(id)path {
  return [super init];
}
-(instancetype) init {
  return [self initWithBasePath:0];
}
@end

@interface Test1 : NSObject
-(instancetype) init NS_DESIGNATED_INITIALIZER;
@end
@implementation Test1
-(instancetype) init {
  return self;
}
@end

@interface SubTest1 : Test1
-(instancetype)init NS_UNAVAILABLE;
-(instancetype)initWithRequiredParameter:(id)foo NS_DESIGNATED_INITIALIZER;
@end
@implementation SubTest1
-(instancetype)initWithRequiredParameter:(id)foo {
  return [super init];
}
@end

@interface SubTest1Ext : Test1
-(instancetype)initWithRequiredParameter:(id)foo NS_DESIGNATED_INITIALIZER;
@end
// Mark 'init' as unavailable in the extension to silence warning.
@interface SubTest1Ext()
-(instancetype)init NS_UNAVAILABLE;
@end
@implementation SubTest1Ext
-(instancetype)initWithRequiredParameter:(id)foo {
  return [super init];
}
@end

@interface Test2 : NSObject
@end
@interface SubTest2 : Test2
@end
@implementation SubTest2
-(instancetype) init { // expected-warning {{designated initializer missing a 'super' call to a designated initializer of the super class}}
  return self;
}
@end

__attribute__((objc_root_class))
@interface RootNoDI
-(id)init;
@end

@interface Base : RootNoDI
@end

@implementation Base
@end

@interface Derived : Base
- (instancetype)initWithInt:(int)n NS_DESIGNATED_INITIALIZER;
@end

@implementation Derived
- (instancetype)initWithInt:(int)n
{
  return [super init];
}
@end

@interface ExtensionForMissingInterface() // expected-error{{cannot find interface declaration}}
- (instancetype)init NS_DESIGNATED_INITIALIZER;
@end

@interface CategoryForMissingInterface(Cat) // expected-error{{cannot find interface declaration}}
- (instancetype)init NS_DESIGNATED_INITIALIZER; // expected-error{{only applies to init methods of interface or class extension declarations}}
@end

@interface TwoAttrs
-(instancetype)foo
    __attribute__((objc_designated_initializer))
    __attribute__((objc_method_family(init)));
-(instancetype)bar
    __attribute__((objc_method_family(init)))
    __attribute__((objc_designated_initializer));
-(instancetype)baz
  __attribute__((objc_designated_initializer, objc_method_family(init)));
-(instancetype)quux
  __attribute__((objc_method_family(init), objc_designated_initializer));
@end
