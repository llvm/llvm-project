// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,osx.cocoa.Dealloc,debug.ExprInspection -analyzer-store=region -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,osx.cocoa.Dealloc,debug.ExprInspection -analyzer-store=region -verify -Wno-objc-root-class -fobjc-arc %s

void clang_analyzer_eval(int);

typedef const void * CFTypeRef;
extern CFTypeRef CFRetain(CFTypeRef cf);
void CFRelease(CFTypeRef cf);

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
-(oneway void)release;
-(void)dealloc;
@end
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
-(id)initWithFormat:(NSString *)f,...;
-(BOOL)isEqualToString:(NSString *)s;
+ (id)string;
@end
@interface NSNumber : NSObject {}
+(id)alloc;
-(id)initWithInteger:(int)i;
@end

// rdar://6946338

@interface Test1 : NSObject {
  NSString *text;
}
-(id)myMethod;
@property (nonatomic, assign) NSString *text;
@end


#if !__has_feature(objc_arc)

@implementation Test1

@synthesize text;

-(id)myMethod {
  Test1 *cell = [[[Test1 alloc] init] autorelease];

  NSString *string1 = [[NSString alloc] initWithFormat:@"test %f", 0.0]; // expected-warning {{Potential leak}}
  cell.text = string1;

  return cell;
}

@end


// rdar://8824416

@interface MyNumber : NSObject
{
  NSNumber* _myNumber;
}

- (id)initWithNumber:(NSNumber *)number;

@property (nonatomic, readonly) NSNumber* myNumber;
@property (nonatomic, readonly) NSNumber* newMyNumber;

@end

@implementation MyNumber
@synthesize myNumber=_myNumber;
 
- (id)initWithNumber:(NSNumber *)number
{
  self = [super init];
  
  if ( self )
  {
    _myNumber = [number copy];
  }
  
  return self;
}

- (NSNumber*)newMyNumber
{
  if ( _myNumber )
    return [_myNumber retain];
  
  return [[NSNumber alloc] initWithInteger:1];
}

- (id)valueForUndefinedKey:(NSString*)key
{
  id value = 0;
  
  if ([key isEqualToString:@"MyIvarNumberAsPropertyOverReleased"])
    value = self.myNumber; // _myNumber will be over released, since the value returned from self.myNumber is not retained.
  else if ([key isEqualToString:@"MyIvarNumberAsPropertyOk"])
    value = [self.myNumber retain]; // this line fixes the over release
  else if ([key isEqualToString:@"MyIvarNumberAsNewMyNumber"])
    value = self.newMyNumber; // this one is ok, since value is returned retained
  else 
    value = [[NSNumber alloc] initWithInteger:0];
  
  return [value autorelease]; // expected-warning {{Object autoreleased too many times}}
}

@end

NSNumber* numberFromMyNumberProperty(MyNumber* aMyNumber)
{
  NSNumber* result = aMyNumber.myNumber;
    
  return [result autorelease]; // expected-warning {{Object autoreleased too many times}}
}

#endif


// rdar://6611873

@interface Person : NSObject {
  NSString *_name;
}
@property (retain) NSString * name;
@property (assign) id friend;
@end

@implementation Person
@synthesize name = _name;

-(void)dealloc {
#if !__has_feature(objc_arc)
  self.name = [[NSString alloc] init]; // expected-warning {{leak}}

  [super dealloc]; // expected-warning {{The '_name' ivar in 'Person' was retained by a synthesized property but not released before '[super dealloc]}}
#endif
}
@end

#if !__has_feature(objc_arc)
void rdar6611873() {
  Person *p = [[[Person alloc] init] autorelease];
  
  p.name = [[NSString string] retain]; // expected-warning {{leak}}
  p.name = [[NSString alloc] init]; // expected-warning {{leak}}

  p.friend = [[Person alloc] init]; // expected-warning {{leak}}
}
#endif

@interface SubPerson : Person
-(NSString *)foo;
@end

@implementation SubPerson
-(NSString *)foo {
  return super.name;
}
@end


#if !__has_feature(objc_arc)
// <rdar://problem/9241180> Static analyzer doesn't detect uninitialized variable issues for property accesses
@interface RDar9241180
@property (readwrite,assign) id x;
-(id)testAnalyzer1:(int) y;
-(void)testAnalyzer2;
@end

@implementation RDar9241180
@synthesize x;
-(id)testAnalyzer1:(int)y {
    RDar9241180 *o;
    if (y && o.x) // expected-warning {{Property access on an uninitialized object pointer}}
      return o;
    return o; // expected-warning {{Undefined or garbage value returned to caller}}
}
-(void)testAnalyzer2 {
  id y;
  self.x = y;  // expected-warning {{Argument for property setter is an uninitialized value}}
}
@end
#endif


//------
// Property accessor synthesis
//------

extern void doSomethingWithPerson(Person *p);
extern void doSomethingWithName(NSString *name);

void testConsistencyRetain(Person *p) {
  clang_analyzer_eval(p.name == p.name); // expected-warning{{TRUE}}

  id origName = p.name;
  clang_analyzer_eval(p.name == origName); // expected-warning{{TRUE}}
  doSomethingWithPerson(p);
  clang_analyzer_eval(p.name == origName); // expected-warning{{UNKNOWN}}
}

void testConsistencyAssign(Person *p) {
  clang_analyzer_eval(p.friend == p.friend); // expected-warning{{TRUE}}

  id origFriend = p.friend;
  clang_analyzer_eval(p.friend == origFriend); // expected-warning{{TRUE}}
  doSomethingWithPerson(p);
  clang_analyzer_eval(p.friend == origFriend); // expected-warning{{UNKNOWN}}
}

@interface ClassWithShadowedReadWriteProperty {
  int _f;
}
@property (readonly) int someProp;
@end

@interface ClassWithShadowedReadWriteProperty ()
@property (readwrite) int someProp;
@end

@implementation ClassWithShadowedReadWriteProperty
- (void)testSynthesisForShadowedReadWriteProperties; {
  clang_analyzer_eval(self.someProp == self.someProp); // expected-warning{{TRUE}}

  _f = 1;

  // Read of shadowed property should not invalidate receiver.
  (void)self.someProp;
  clang_analyzer_eval(_f == 1); // expected-warning{{TRUE}}

  _f = 2;
  // Call to getter of shadowed property should not invalidate receiver.
  (void)[self someProp];
  clang_analyzer_eval(_f == 2); // expected-warning{{TRUE}}
}
@end

// Tests for the analyzer fix that works around a Sema bug
// where multiple methods are created for properties in class extensions that
// are redeclared in a category method.
// The Sema bug is tracked as <rdar://problem/25481164>.
@interface ClassWithRedeclaredPropertyInExtensionFollowedByCategory
@end

@interface ClassWithRedeclaredPropertyInExtensionFollowedByCategory ()
@end

@interface ClassWithRedeclaredPropertyInExtensionFollowedByCategory ()
@property (readwrite) int someProp;
@property (readonly) int otherProp;
@end

@interface ClassWithRedeclaredPropertyInExtensionFollowedByCategory (MyCat)
@property (readonly) int someProp;
@property (readonly) int otherProp;
@end

@implementation ClassWithRedeclaredPropertyInExtensionFollowedByCategory
- (void)testSynthesisForRedeclaredProperties; {
  clang_analyzer_eval(self.someProp == self.someProp); // expected-warning{{TRUE}}
  clang_analyzer_eval([self someProp] == self.someProp); // expected-warning{{TRUE}}

  clang_analyzer_eval(self.otherProp == self.otherProp); // expected-warning{{TRUE}}
  clang_analyzer_eval([self otherProp] == self.otherProp); // expected-warning{{TRUE}}
}
@end

// The relative order of the extension and the category matter, so test both.
@interface ClassWithRedeclaredPropertyInCategoryFollowedByExtension
@end

@interface ClassWithRedeclaredPropertyInCategoryFollowedByExtension ()
@property (readwrite) int someProp;
@end

@interface ClassWithRedeclaredPropertyInCategoryFollowedByExtension (MyCat)
@property (readonly) int someProp;
@end

@implementation ClassWithRedeclaredPropertyInCategoryFollowedByExtension
- (void)testSynthesisForRedeclaredProperties; {
  clang_analyzer_eval(self.someProp == self.someProp); // expected-warning{{TRUE}}
  clang_analyzer_eval([self someProp] == self.someProp); // expected-warning{{TRUE}}
}
@end

@interface ClassWithSynthesizedPropertyAndGetter
@property (readonly) int someProp;
@end

@implementation ClassWithSynthesizedPropertyAndGetter
@synthesize someProp;

// Make sure that the actual getter is inlined and not a getter created
// by BodyFarm
- (void)testBodyFarmGetterNotUsed {
  int i = self.someProp;
  clang_analyzer_eval(i == 22); // expected-warning {{TRUE}}
}

-(int)someProp {
  return 22;
}
@end

__attribute__((objc_root_class))
@interface ClassWithPrivatePropertyInClassExtensionWithProtocolShadowingCategory
@end

@protocol HasStuff
@property (nonatomic, readonly) int stuffProperty;
@end

@interface ClassWithPrivatePropertyInClassExtensionWithProtocolShadowingCategory (Private)
@property (nonatomic, readonly) int stuffProperty;
@end

@interface ClassWithPrivatePropertyInClassExtensionWithProtocolShadowingCategory (Internal) <HasStuff>
@end

@interface ClassWithPrivatePropertyInClassExtensionWithProtocolShadowingCategory() <HasStuff>
@end

@implementation ClassWithPrivatePropertyInClassExtensionWithProtocolShadowingCategory
@synthesize stuffProperty = _stuffProperty;

-(void)foo {
  (void)self.stuffProperty;
}
@end

//------
// Setter ivar invalidation.
//------

@interface ClassWithSetters
// Note: These properties have implicit @synthesize implementations to be
// backed with ivars.
@property (assign) int propWithIvar1;
@property (assign) int propWithIvar2;

@property (retain) NSNumber *retainedProperty;

@end

@interface ClassWithSetters (InOtherTranslationUnit)
// The implementation of this property is in another translation unit.
// We don't know whether it is backed by an ivar or not.
@property (assign) int propInOther;
@end

@implementation ClassWithSetters
- (void) testSettingPropWithIvarInvalidatesExactlyThatIvar; {
  _propWithIvar1 = 1;
  _propWithIvar2 = 2;
  self.propWithIvar1 = 66;

  // Calling the setter of a property backed by the instance variable
  // should invalidate the storage for the instance variable but not
  // the rest of the receiver. Ideally we would model the setter completely
  // but doing so would cause the new value to escape when it is bound
  // to the ivar. This would cause bad false negatives in the retain count
  // checker. (There is a test for this scenario in
  // testWriteRetainedValueToRetainedProperty below).
  clang_analyzer_eval(_propWithIvar1 == 66); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(_propWithIvar2 == 2);  // expected-warning{{TRUE}}

  _propWithIvar1 = 1;
  [self setPropWithIvar1:66];

  clang_analyzer_eval(_propWithIvar1 == 66); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(_propWithIvar2 == 2);  // expected-warning{{TRUE}}
}

- (void) testSettingPropWithoutIvarInvalidatesEntireInstance; {
  _propWithIvar1 = 1;
  _propWithIvar2 = 2;
  self.propInOther = 66;

  clang_analyzer_eval(_propWithIvar1 == 66); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(_propWithIvar2 == 2);  // expected-warning{{UNKNOWN}}

  _propWithIvar1 = 1;
  _propWithIvar2 = 2;
  [self setPropInOther:66];

  clang_analyzer_eval(_propWithIvar1 == 66); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(_propWithIvar2 == 2);  // expected-warning{{UNKNOWN}}
}

#if !__has_feature(objc_arc)
- (void) testWriteRetainedValueToRetainedProperty; {
  NSNumber *number = [[NSNumber alloc] initWithInteger:5]; // expected-warning {{Potential leak of an object stored into 'number'}}

  // Make sure we catch this leak.
  self.retainedProperty = number;
}
#endif
@end

//------
// class properties
//------

int gBackingForReadWriteClassProp = 0;

@interface ClassWithClassProperties
@property(class, readonly) int readOnlyClassProp;

@property(class) int readWriteClassProp;

// Make sure we handle when a class and instance property have the same
// name. Test both when instance comes first and when class comes first.
@property(readonly) int classAndInstancePropWithSameNameOrderInstanceFirst;
@property(class, readonly) int classAndInstancePropWithSameNameOrderInstanceFirst;

@property(class, readonly) int classAndInstancePropWithSameNameOrderClassFirst;
@property(readonly) int classAndInstancePropWithSameNameOrderClassFirst;


@property(class, readonly) int dynamicClassProp;

@end

@interface ClassWithClassProperties (OtherTranslationUnit)
@property(class, assign) id propInOtherTranslationUnit;
@end

@implementation ClassWithClassProperties

@dynamic dynamicClassProp;

+ (int)readOnlyClassProp {
  return 1;
}

+ (int)readWriteClassProp {
  return gBackingForReadWriteClassProp;
}

+ (void)setReadWriteClassProp:(int)val {
  gBackingForReadWriteClassProp = val;
}

- (int)classAndInstancePropWithSameNameOrderInstanceFirst {
  return 12;
}

+ (int)classAndInstancePropWithSameNameOrderInstanceFirst {
  return 13;
}

+ (int)classAndInstancePropWithSameNameOrderClassFirst {
  return 14;
}

- (int)classAndInstancePropWithSameNameOrderClassFirst {
  return 15;
}

- (void)testInlineClassProp {
  clang_analyzer_eval(ClassWithClassProperties.readOnlyClassProp == 1); // expected-warning{{TRUE}}

  ClassWithClassProperties.readWriteClassProp = 7;
  clang_analyzer_eval(ClassWithClassProperties.readWriteClassProp == 7); // expected-warning{{TRUE}}
  ClassWithClassProperties.readWriteClassProp = 8;
  clang_analyzer_eval(ClassWithClassProperties.readWriteClassProp == 8); // expected-warning{{TRUE}}
}

- (void)testUnknownClassProp {
  clang_analyzer_eval(ClassWithClassProperties.propInOtherTranslationUnit == ClassWithClassProperties.propInOtherTranslationUnit); // expected-warning{{UNKNOWN}}
}

- (void)testEscapeGlobalOnUnknownProp {
  gBackingForReadWriteClassProp = 33;
  ClassWithClassProperties.propInOtherTranslationUnit = 0;
  clang_analyzer_eval(gBackingForReadWriteClassProp == 33); // expected-warning{{UNKNOWN}}
}

- (void)testClassAndInstancePropertyWithSameName {
  clang_analyzer_eval(self.classAndInstancePropWithSameNameOrderInstanceFirst == 12); // expected-warning{{TRUE}}
  clang_analyzer_eval(ClassWithClassProperties.classAndInstancePropWithSameNameOrderInstanceFirst == 13); // expected-warning{{TRUE}}

  clang_analyzer_eval(ClassWithClassProperties.classAndInstancePropWithSameNameOrderClassFirst == 14); // expected-warning{{TRUE}}
  clang_analyzer_eval(self.classAndInstancePropWithSameNameOrderClassFirst == 15); // expected-warning{{TRUE}}
}

- (void)testDynamicClassProp {
  clang_analyzer_eval(ClassWithClassProperties.dynamicClassProp == 16); // expected-warning{{UNKNOWN}}
}

@end

@interface SubclassOfClassWithClassProperties : ClassWithClassProperties
@end

@implementation SubclassOfClassWithClassProperties
+ (int)dynamicClassProp; {
 return 16;
}

- (void)testDynamicClassProp {
  clang_analyzer_eval(SubclassOfClassWithClassProperties.dynamicClassProp == 16); // expected-warning{{TRUE}}
}

@end


#if !__has_feature(objc_arc)
void testOverrelease(Person *p, int coin) {
  switch (coin) {
  case 0:
    [p.name release]; // expected-warning{{not owned}}
    break;
  case 1:
    [p.friend release]; // expected-warning{{not owned}}
    break;
  case 2: {
    id friend = p.friend;
    doSomethingWithPerson(p);
    [friend release]; // expected-warning{{not owned}}
  }
  }
}

// <rdar://problem/16333368>
@implementation Person (Rdar16333368)

- (void)testDeliberateRelease:(Person *)other {
  doSomethingWithName(self.name);
  [_name release]; // no-warning
  self->_name = 0;

  doSomethingWithName(other->_name);
  [other.name release]; // no-warning
}

- (void)deliberateReleaseFalseNegative {
  // This is arguably a false negative because the result of p.friend shouldn't
  // be released, even though we are manipulating the ivar in between the two
  // actions.
  id name = self.name;
  _name = 0;
  [name release];
}

- (void)testRetainAndRelease {
  [self.name retain];
  [self.name release];
  [self.name release]; // expected-warning{{not owned}}
}

- (void)testRetainAndReleaseIVar {
  [self.name retain];
  [_name release];
  [_name release];
}

@end
#endif

@interface IntWrapper
@property int value;
@end

@implementation IntWrapper
@synthesize value;
@end

void testConsistencyInt(IntWrapper *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{TRUE}}

  int origValue = w.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}

void testConsistencyInt2(IntWrapper *w) {
  if (w.value != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}


@interface IntWrapperAuto
@property int value;
@end

@implementation IntWrapperAuto
@end

void testConsistencyIntAuto(IntWrapperAuto *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{TRUE}}

  int origValue = w.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}

void testConsistencyIntAuto2(IntWrapperAuto *w) {
  if (w.value != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}


typedef struct {
  int value;
} IntWrapperStruct;

@interface StructWrapper
@property IntWrapperStruct inner;
@end

@implementation StructWrapper
@synthesize inner;
@end

void testConsistencyStruct(StructWrapper *w) {
  clang_analyzer_eval(w.inner.value == w.inner.value); // expected-warning{{TRUE}}

  int origValue = w.inner.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.inner.value == 42); // expected-warning{{TRUE}}
}


@interface OpaqueIntWrapper
@property int value;
@end

// For now, don't assume a property is implemented using an ivar unless we can
// actually see that it is.
void testOpaqueConsistency(OpaqueIntWrapper *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{UNKNOWN}}
}


#if !__has_feature(objc_arc)
// Test quite a few cases of retain/release issues.

@interface RetainCountTesting
@property (strong) id ownedProp;
@property (unsafe_unretained) id unownedProp;
@property (nonatomic, strong) id manualProp;
@property (readonly) id readonlyProp;
@property (nonatomic, readwrite/*, assign */) id implicitManualProp; // expected-warning {{'assign' is assumed}} expected-warning {{'assign' not appropriate}}
@property (nonatomic, readwrite/*, assign */) id implicitSynthProp; // expected-warning {{'assign' is assumed}} expected-warning {{'assign' not appropriate}}
@property CFTypeRef cfProp;
@end

@implementation RetainCountTesting {
  id _ivarOnly;
}

- (id)manualProp {
  return _manualProp;
}

- (void)setImplicitManualProp:(id)newValue {}

- (void)testOverreleaseOwnedIvar {
  [_ownedProp retain];
  [_ownedProp release];
  [_ownedProp release];
  [_ownedProp release]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseUnownedIvar {
  [_unownedProp retain];
  [_unownedProp release];
  [_unownedProp release]; // FIXME-warning{{not owned at this point by the caller}}
}

- (void)testOverreleaseIvarOnly {
  [_ivarOnly retain];
  [_ivarOnly release];
  [_ivarOnly release];
  [_ivarOnly release]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseReadonlyIvar {
  [_readonlyProp retain];
  [_readonlyProp release];
  [_readonlyProp release];
  [_readonlyProp release]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseImplicitManualIvar {
  [_implicitManualProp retain];
  [_implicitManualProp release];
  [_implicitManualProp release];
  [_implicitManualProp release]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseImplicitSynthIvar {
  [_implicitSynthProp retain];
  [_implicitSynthProp release];
  [_implicitSynthProp release]; // FIXME-warning{{not owned at this point by the caller}}
}

- (void)testOverreleaseCF {
  CFRetain(_cfProp);
  CFRelease(_cfProp);
  CFRelease(_cfProp);
  CFRelease(_cfProp); // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseOwnedIvarUse {
  [_ownedProp retain];
  [_ownedProp release];
  [_ownedProp release];
  [_ownedProp myMethod]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseIvarOnlyUse {
  [_ivarOnly retain];
  [_ivarOnly release];
  [_ivarOnly release];
  [_ivarOnly myMethod]; // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseCFUse {
  CFRetain(_cfProp);
  CFRelease(_cfProp);
  CFRelease(_cfProp);

  extern void CFUse(CFTypeRef);
  CFUse(_cfProp); // FIXME-warning{{used after it is released}}
}

- (void)testOverreleaseOwnedIvarAutoreleaseOkay {
  [_ownedProp retain];
  [_ownedProp release];
  [_ownedProp autorelease];
} // no-warning

- (void)testOverreleaseIvarOnlyAutoreleaseOkay {
  [_ivarOnly retain];
  [_ivarOnly release];
  [_ivarOnly autorelease];
} // no-warning

- (void)testOverreleaseOwnedIvarAutorelease {
  [_ownedProp retain];
  [_ownedProp release];
  [_ownedProp autorelease];
  [_ownedProp autorelease];
} // FIXME-warning{{Object autoreleased too many times}}

- (void)testOverreleaseIvarOnlyAutorelease {
  [_ivarOnly retain];
  [_ivarOnly release];
  [_ivarOnly autorelease];
  [_ivarOnly autorelease];
} // FIXME-warning{{Object autoreleased too many times}}

- (void)testPropertyAccessThenReleaseOwned {
  id owned = [self.ownedProp retain];
  [owned release];
  [_ownedProp release];
  clang_analyzer_eval(owned == _ownedProp); // expected-warning{{TRUE}}
}

- (void)testPropertyAccessThenReleaseOwned2 {
  id fromIvar = _ownedProp;
  id owned = [self.ownedProp retain];
  [owned release];
  [fromIvar release];
  clang_analyzer_eval(owned == fromIvar); // expected-warning{{TRUE}}
}

- (void)testPropertyAccessThenReleaseUnowned {
  id unowned = [self.unownedProp retain];
  [unowned release];
  [_unownedProp release]; // FIXME-warning{{not owned}}
}

- (void)testPropertyAccessThenReleaseUnowned2 {
  id fromIvar = _unownedProp;
  id unowned = [self.unownedProp retain];
  [unowned release];
  clang_analyzer_eval(unowned == fromIvar); // expected-warning{{TRUE}}
  [fromIvar release]; // FIXME-warning{{not owned}}
}

- (void)testPropertyAccessThenReleaseManual {
  id prop = [self.manualProp retain];
  [prop release];
  [_manualProp release]; // no-warning
}

- (void)testPropertyAccessThenReleaseManual2 {
  id fromIvar = _manualProp;
  id prop = [self.manualProp retain];
  [prop release];
  clang_analyzer_eval(prop == fromIvar); // expected-warning{{TRUE}}
  [fromIvar release]; // no-warning
}

- (void)testPropertyAccessThenReleaseCF {
  CFTypeRef owned = CFRetain(self.cfProp);
  CFRelease(owned);
  CFRelease(_cfProp); // no-warning
  clang_analyzer_eval(owned == _cfProp); // expected-warning{{TRUE}}
}

- (void)testPropertyAccessThenReleaseCF2 {
  CFTypeRef fromIvar = _cfProp;
  CFTypeRef owned = CFRetain(self.cfProp);
  CFRelease(owned);
  CFRelease(fromIvar);
  clang_analyzer_eval(owned == fromIvar); // expected-warning{{TRUE}}
}

- (void)testPropertyAccessThenReleaseReadonly {
  id prop = [self.readonlyProp retain];
  [prop release];
  [_readonlyProp release]; // no-warning
}

- (void)testPropertyAccessThenReleaseReadonly2 {
  id fromIvar = _readonlyProp;
  id prop = [self.readonlyProp retain];
  [prop release];
  clang_analyzer_eval(prop == fromIvar); // expected-warning{{TRUE}}
  [fromIvar release]; // no-warning
}

- (void)testPropertyAccessThenReleaseImplicitManual {
  id prop = [self.implicitManualProp retain];
  [prop release];
  [_implicitManualProp release]; // no-warning
}

- (void)testPropertyAccessThenReleaseImplicitManual2 {
  id fromIvar = _implicitManualProp;
  id prop = [self.implicitManualProp retain];
  [prop release];
  clang_analyzer_eval(prop == fromIvar); // expected-warning{{TRUE}}
  [fromIvar release]; // no-warning
}

- (void)testPropertyAccessThenReleaseImplicitSynth {
  id prop = [self.implicitSynthProp retain];
  [prop release];
  [_implicitSynthProp release]; // FIXME-warning{{not owned}}
}

- (void)testPropertyAccessThenReleaseImplicitSynth2 {
  id fromIvar = _implicitSynthProp;
  id prop = [self.implicitSynthProp retain];
  [prop release];
  clang_analyzer_eval(prop == fromIvar); // expected-warning{{TRUE}}
  [fromIvar release]; // FIXME-warning{{not owned}}
}

- (id)getUnownedFromProperty {
  [_ownedProp retain];
  [_ownedProp autorelease];
  return _ownedProp; // no-warning
}

- (id)transferUnownedFromProperty {
  [_ownedProp retain];
  [_ownedProp autorelease];
  return [_ownedProp autorelease]; // no-warning
}

- (id)transferOwnedFromProperty __attribute__((ns_returns_retained)) {
  [_ownedProp retain];
  [_ownedProp autorelease];
  return _ownedProp; // no-warning
}

- (void)testAssignOwned:(id)newValue {
  _ownedProp = newValue;
  [_ownedProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignUnowned:(id)newValue {
  _unownedProp = newValue;
  [_unownedProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignIvarOnly:(id)newValue {
  _ivarOnly = newValue;
  [_ivarOnly release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignCF:(CFTypeRef)newValue {
  _cfProp = newValue;
  CFRelease(_cfProp); // FIXME: no-warning{{not owned}}
}

- (void)testAssignReadonly:(id)newValue {
  _readonlyProp = newValue;
  [_readonlyProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignImplicitManual:(id)newValue {
  _implicitManualProp = newValue;
  [_implicitManualProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignImplicitSynth:(id)newValue {
  _implicitSynthProp = newValue;
  [_implicitSynthProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignOwnedOkay:(id)newValue {
  _ownedProp = [newValue retain];
  [_ownedProp release]; // no-warning
}

- (void)testAssignUnownedOkay:(id)newValue {
  _unownedProp = [newValue retain];
  [_unownedProp release]; // no-warning
}

- (void)testAssignIvarOnlyOkay:(id)newValue {
  _ivarOnly = [newValue retain];
  [_ivarOnly release]; // no-warning
}

- (void)testAssignCFOkay:(CFTypeRef)newValue {
  _cfProp = CFRetain(newValue);
  CFRelease(_cfProp); // no-warning
}

- (void)testAssignReadonlyOkay:(id)newValue {
  _readonlyProp = [newValue retain];
  [_readonlyProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignImplicitManualOkay:(id)newValue {
  _implicitManualProp = [newValue retain];
  [_implicitManualProp release]; // FIXME: no-warning{{not owned}}
}

- (void)testAssignImplicitSynthOkay:(id)newValue {
  _implicitSynthProp = [newValue retain];
  [_implicitSynthProp release]; // FIXME: no-warning{{not owned}}
}

// rdar://problem/19862648
- (void)establishIvarIsNilDuringLoops {
  extern id getRandomObject();

  int i = 4; // Must be at least 4 to trigger the bug.
  while (--i) {
    id x = 0;
    if (getRandomObject())
      x = _ivarOnly;
    if (!x)
      x = getRandomObject();
    [x myMethod];
  }
}

// rdar://problem/20335433
- (void)retainIvarAndInvalidateSelf {
  extern void invalidate(id);
  [_unownedProp retain];
  invalidate(self);
  [_unownedProp release]; // no-warning
}

@end

@interface Wrapper
@property(nonatomic, readonly) int value;
@end

@implementation Wrapper
@synthesize value;
@end

void testNoCrashWhenAccessPropertyAndThereAreNoDirectBindingsAtAll() {
   union {
    Wrapper *wrapper;
   } u = { 0 };
   [u.wrapper value];
}

#endif // non-ARC

