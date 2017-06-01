// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.Dealloc -fblocks -triple x86_64-apple-ios4.0 -DMACOS=0 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.Dealloc -fblocks -triple x86_64-apple-macosx10.6.0 -DMACOS=1 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.Dealloc -fblocks -triple x86_64-apple-darwin10 -fobjc-arc -fobjc-runtime-has-weak -verify %s

#include "Inputs/system-header-simulator-for-objc-dealloc.h"

#define NON_ARC !__has_feature(objc_arc)

#if NON_ARC
#define WEAK_ON_ARC
#else
#define WEAK_ON_ARC __weak
#endif

// No diagnostics expected under ARC.
#if !NON_ARC
  // expected-no-diagnostics
#endif

// Do not warn about missing release in -dealloc for ivars.

@interface MyIvarClass1 : NSObject {
  NSObject *_ivar;
}
@end

@implementation MyIvarClass1
- (instancetype)initWithIvar:(NSObject *)ivar
{
  self = [super init];
  if (!self)
    return nil;
#if NON_ARC
  _ivar = [ivar retain];
#endif
  return self;
}
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface MyIvarClass2 : NSObject {
  NSObject *_ivar;
}
- (NSObject *)ivar;
- (void)setIvar:(NSObject *)ivar;
@end

@implementation MyIvarClass2
- (instancetype)init
{
  self = [super init];
  return self;
}
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
- (NSObject *)ivar
{
  return _ivar;
}
- (void)setIvar:(NSObject *)ivar
{
#if NON_ARC
  [_ivar release];
  _ivar = [ivar retain];
#else
 _ivar = ivar;
#endif
}
@end

// Warn about missing release in -dealloc for properties.

@interface MyPropertyClass1 : NSObject
@property (copy) NSObject *ivar;
@end

@implementation MyPropertyClass1
- (void)dealloc
{
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'MyPropertyClass1' was copied by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface MyPropertyClass2 : NSObject
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClass2
- (void)dealloc
{
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'MyPropertyClass2' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface MyPropertyClass3 : NSObject {
  NSObject *_ivar;
}
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClass3
@synthesize ivar = _ivar;
- (void)dealloc
{
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'MyPropertyClass3' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}

@end

@interface MyPropertyClass4 : NSObject {
  void (^_blockPropertyIvar)(void);
}
@property (copy) void (^blockProperty)(void);
@property (copy) void (^blockProperty2)(void);
@property (copy) void (^blockProperty3)(void);

@end

@implementation MyPropertyClass4
@synthesize blockProperty = _blockPropertyIvar;
- (void)dealloc
{
#if NON_ARC
  [_blockProperty2 release];
  Block_release(_blockProperty3);

  [super dealloc]; // expected-warning {{The '_blockPropertyIvar' ivar in 'MyPropertyClass4' was copied by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface MyPropertyClass5 : NSObject {
  WEAK_ON_ARC NSObject *_ivar;
}
@property (weak) NSObject *ivar;
@end

@implementation MyPropertyClass5
@synthesize ivar = _ivar;
- (void)dealloc
{
#if NON_ARC
  [super dealloc]; // no-warning because it is a weak property
#endif
}
@end

@interface MyPropertyClassWithReturnInDealloc : NSObject {
  NSObject *_ivar;
}
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClassWithReturnInDealloc
@synthesize ivar = _ivar;
- (void)dealloc
{
  return;
#if NON_ARC
  // expected-warning@-2{{The '_ivar' ivar in 'MyPropertyClassWithReturnInDealloc' was retained by a synthesized property but not released before '[super dealloc]'}}
  [super dealloc];
#endif
}
@end

@interface MyPropertyClassWithReleaseInOtherInstance : NSObject {
  NSObject *_ivar;
  MyPropertyClassWithReleaseInOtherInstance *_other;
}
@property (retain) NSObject *ivar;

-(void)releaseIvars;
@end

@implementation MyPropertyClassWithReleaseInOtherInstance
@synthesize ivar = _ivar;

-(void)releaseIvars; {
#if NON_ARC
  [_ivar release];
#endif
}

- (void)dealloc
{
  [_other releaseIvars];
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'MyPropertyClassWithReleaseInOtherInstance' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface MyPropertyClassWithNeitherReturnNorSuperDealloc : NSObject {
  NSObject *_ivar;
}
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClassWithNeitherReturnNorSuperDealloc
@synthesize ivar = _ivar;
- (void)dealloc
{
}
#if NON_ARC
  // expected-warning@-2 {{method possibly missing a [super dealloc] call}} (From Sema)
  // expected-warning@-3{{The '_ivar' ivar in 'MyPropertyClassWithNeitherReturnNorSuperDealloc' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
@end

// <rdar://problem/6380411>: 'myproperty' has kind 'assign' and thus the
//  assignment through the setter does not perform a release.

@interface MyObject : NSObject {
  id __unsafe_unretained _myproperty;
}
@property(assign) id myproperty;
@end

@implementation MyObject
@synthesize myproperty=_myproperty; // no-warning
- (void)dealloc {
  // Don't claim that myproperty is released since it the property
  // has the 'assign' attribute.
  self.myproperty = 0; // no-warning
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface ClassWithControlFlowInRelease : NSObject {
  BOOL _ivar1;
}
@property (retain) NSObject *ivar2;
@end

@implementation ClassWithControlFlowInRelease
- (void)dealloc; {
  if (_ivar1) {
    // We really should warn because there is a path through -dealloc on which
    // _ivar2 is not released.
#if NON_ARC
    [_ivar2 release];
#endif
  }

#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar2' ivar in 'ClassWithControlFlowInRelease' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

// Don't warn when the property is nil'd out in -dealloc

@interface ClassWithNildOutProperty : NSObject
@property (retain) NSObject *ivar;
@property (assign) int *intPtrProp;
@end

@implementation ClassWithNildOutProperty
- (void)dealloc; {
  self.ivar = nil;

  // Make sure to handle setting a non-retainable property to 0.
  self.intPtrProp = 0;
#if NON_ARC
  [super dealloc];  // no-warning
#endif
}
@end

// Do warn when the ivar but not the property is nil'd out in -dealloc

@interface ClassWithNildOutIvar : NSObject
@property (retain) NSObject *ivar;
@end

@implementation ClassWithNildOutIvar
- (void)dealloc; {
  // Oops. Meant self.ivar = nil
  _ivar = nil;

#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'ClassWithNildOutIvar' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

// Do warn when the ivar is updated to a different value that is then
// released.

@interface ClassWithUpdatedIvar : NSObject
@property (retain) NSObject *ivar;
@end

@implementation ClassWithUpdatedIvar
- (void)dealloc; {
  _ivar = [[NSObject alloc] init];

#if NON_ARC
  [_ivar release];
#endif

#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivar' ivar in 'ClassWithUpdatedIvar' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end


// Don't warn when the property is nil'd out with a setter in -dealloc

@interface ClassWithNildOutPropertyViaSetter : NSObject
@property (retain) NSObject *ivar;
@end

@implementation ClassWithNildOutPropertyViaSetter
- (void)dealloc; {
  [self setIvar:nil];

#if NON_ARC
  [super dealloc];  // no-warning
#endif
}
@end


// Don't warn about missing releases when -dealloc helpers are called.

@interface ClassWithDeallocHelpers : NSObject
@property (retain) NSObject *ivarReleasedInMethod;
@property (retain) NSObject *propNilledOutInMethod;

@property (retain) NSObject *ivarReleasedInFunction;
@property (retain) NSObject *propNilledOutInFunction;

@property (retain) NSObject *ivarNeverReleased;
- (void)invalidateInMethod;
@end

void ReleaseValueHelper(NSObject *iv) {
#if NON_ARC
  [iv release];
#endif
}

void NilOutPropertyHelper(ClassWithDeallocHelpers *o) {
  o.propNilledOutInFunction = nil;
}

@implementation ClassWithDeallocHelpers
- (void)invalidateInMethod {
#if NON_ARC
  [_ivarReleasedInMethod release];
#endif
  self.propNilledOutInMethod = nil;
}

- (void)dealloc; {
  ReleaseValueHelper(_ivarReleasedInFunction);
  NilOutPropertyHelper(self);

  [self invalidateInMethod];
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_ivarNeverReleased' ivar in 'ClassWithDeallocHelpers' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end


// Don't warn when self in -dealloc escapes.

@interface ClassWhereSelfEscapesViaMethodCall : NSObject
@property (retain) NSObject *ivar;  // no-warning
@end

@interface ClassWhereSelfEscapesViaMethodCall (Other)
- (void)invalidate; // In other translation unit.
@end

@implementation ClassWhereSelfEscapesViaMethodCall
- (void)dealloc; {
  [self invalidate];
#if NON_ARC
  [super dealloc];
#endif
} // no-warning
@end

@interface ClassWhereSelfEscapesViaPropertyAccess : NSObject
@property (retain) NSObject *ivar;
@end

@interface ClassWhereSelfEscapesViaPropertyAccess (Other)
// The implementation of this property is unknown and therefore could
// release ivar.
@property (retain) NSObject *otherIvar;
@end

@implementation ClassWhereSelfEscapesViaPropertyAccess
- (void)dealloc; {
  self.otherIvar = nil;
#if NON_ARC
  [super dealloc];
#endif
} // no-warning
@end

// Don't treat self as escaping when setter called on *synthesized*
// property.

@interface ClassWhereSelfEscapesViaSynthesizedPropertyAccess : NSObject
@property (retain) NSObject *ivar;
@property (retain) NSObject *otherIvar;
@end

@implementation ClassWhereSelfEscapesViaSynthesizedPropertyAccess
- (void)dealloc; {
  self.otherIvar = nil;
#if NON_ARC
  [super dealloc];  // expected-warning {{The '_ivar' ivar in 'ClassWhereSelfEscapesViaSynthesizedPropertyAccess' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end


// Don't treat calls to system headers as escapes

@interface ClassWhereSelfEscapesViaCallToSystem : NSObject
@property (retain) NSObject *ivar1;
@property (retain) NSObject *ivar2;
@property (retain) NSObject *ivar3;
@property (retain) NSObject *ivar4;
@property (retain) NSObject *ivar5;
@property (retain) NSObject *ivar6;
@end

@implementation ClassWhereSelfEscapesViaCallToSystem
- (void)dealloc; {
#if NON_ARC
  [_ivar2 release];
  if (_ivar3) {
    [_ivar3 release];
  }
#endif

  [[NSRunLoop currentRunLoop] cancelPerformSelectorsWithTarget:self];
  [[NSNotificationCenter defaultCenter] removeObserver:self];

#if NON_ARC
  [_ivar4 release];

  if (_ivar5) {
    [_ivar5 release];
  }
#endif

  [[NSNotificationCenter defaultCenter] removeObserver:self];

#if NON_ARC
  if (_ivar6) {
    [_ivar6 release];
  }

  [super dealloc];  // expected-warning {{The '_ivar1' ivar in 'ClassWhereSelfEscapesViaCallToSystem' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

// Don't warn when value escapes.

@interface ClassWhereIvarValueEscapes : NSObject
@property (retain) NSObject *ivar;
@end

void ReleaseMe(id arg);

@implementation ClassWhereIvarValueEscapes
- (void)dealloc; {

  ReleaseMe(_ivar);

#if NON_ARC
  [super dealloc];
#endif
} // no-warning
@end

// Don't warn when value is known to be nil.

@interface ClassWhereIvarIsNil : NSObject
@property (retain) NSObject *ivarIsNil;
@end

@implementation ClassWhereIvarIsNil
- (void)dealloc; {

#if NON_ARC
  if (_ivarIsNil)
    [_ivarIsNil release];

  [super dealloc];
#endif
} // no-warning
@end


// Don't warn for non-retainable properties.

@interface ClassWithNonRetainableProperty : NSObject
@property (assign) int *ivar;  // no-warning
@end

@implementation ClassWithNonRetainableProperty
- (void)dealloc; {
#if NON_ARC
  [super dealloc];
#endif
} // no-warning
@end


@interface SuperClassOfClassWithInlinedSuperDealloc : NSObject
@property (retain) NSObject *propInSuper;
@end

@implementation SuperClassOfClassWithInlinedSuperDealloc
- (void)dealloc {
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_propInSuper' ivar in 'SuperClassOfClassWithInlinedSuperDealloc' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface ClassWithInlinedSuperDealloc : SuperClassOfClassWithInlinedSuperDealloc
@property (retain) NSObject *propInSub;
@end

@implementation ClassWithInlinedSuperDealloc
- (void)dealloc {
#if NON_ARC
  [super dealloc]; // expected-warning {{The '_propInSub' ivar in 'ClassWithInlinedSuperDealloc' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end


@interface SuperClassOfClassWithInlinedSuperDeallocAndInvalidation : NSObject
@property (retain) NSObject *propInSuper;

- (void)invalidate;
@end

@implementation SuperClassOfClassWithInlinedSuperDeallocAndInvalidation

- (void)invalidate {
#if NON_ARC
  [_propInSuper release];
#endif
  _propInSuper = nil;
}

- (void)dealloc {
  [self invalidate];
#if NON_ARC
  [super dealloc]; // no-warning
#endif
}
@end

@interface ClassWithInlinedSuperDeallocAndInvalidation : SuperClassOfClassWithInlinedSuperDeallocAndInvalidation
@property (retain) NSObject *propInSub;
@end

@implementation ClassWithInlinedSuperDeallocAndInvalidation

- (void)invalidate {
#if NON_ARC
  [_propInSub release];
#endif
  [super invalidate];
}

- (void)dealloc {
#if NON_ARC
  [super dealloc]; // no-warning
#endif
}
@end


@interface SuperClassOfClassThatEscapesBeforeInliningSuper : NSObject
@property (retain) NSObject *propInSuper;
@end

@implementation SuperClassOfClassThatEscapesBeforeInliningSuper

- (void)dealloc {

#if NON_ARC
  [super dealloc]; // expected-warning {{The '_propInSuper' ivar in 'SuperClassOfClassThatEscapesBeforeInliningSuper' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface ClassThatEscapesBeforeInliningSuper : SuperClassOfClassThatEscapesBeforeInliningSuper
@property (retain) NSObject *propInSub;
@end

@interface ClassThatEscapesBeforeInliningSuper (Other)
- (void)invalidate; // No implementation in translation unit.
@end

@implementation ClassThatEscapesBeforeInliningSuper
- (void)dealloc {
  [self invalidate];

#if NON_ARC
  [super dealloc]; // no-warning
#endif
}
@end


#if NON_ARC
@interface ReleaseIvarInField : NSObject {
  int _tag;
  union {
    NSObject *field1;
    NSObject *field2;
  } _someUnion;

  struct {
    NSObject *field1;
  } _someStruct;
}
@end

@implementation ReleaseIvarInField
- (void)dealloc {
  if (_tag) {
    [_someUnion.field1 release];
  } else {
    [_someUnion.field2 release];
  }

  [_someStruct.field1 release];
  [super dealloc];
}
@end
#endif

struct SomeStruct {
  int f;
};
@interface ZeroOutStructWithSetter : NSObject
  @property(assign) struct SomeStruct s;
@end

@implementation ZeroOutStructWithSetter
- (void)dealloc {
  struct SomeStruct zeroedS;
  zeroedS.f = 0;

  self.s = zeroedS;
#if NON_ARC
  [super dealloc];
#endif
}
@end

#if NON_ARC
@interface ReleaseIvarInArray : NSObject {
  NSObject *_array[3];
}
@end

@implementation ReleaseIvarInArray
- (void)dealloc {
  for (int i = 0; i < 3; i++) {
    [_array[i] release];
  }
  [super dealloc];
}
@end
#endif

// Don't warn about missing releases for subclasses of SenTestCase or
// for classes that are not subclasses of NSObject.

@interface SenTestCase : NSObject {}
@end

@interface MyClassTest : SenTestCase
@property (retain) NSObject *ivar;
@end

@implementation MyClassTest
-(void)tearDown {
#if NON_ARC
  [_ivar release];
#endif
}

-(void)dealloc; {
#if NON_ARC
  [super dealloc]; // no-warning
#endif
}
@end

@interface XCTestCase : NSObject {}
@end

@interface MyClassXCTest : XCTestCase
@property (retain) NSObject *ivar;
@end

@implementation MyClassXCTest
-(void)tearDown {
#if NON_ARC
  [_ivar release];
#endif
}

-(void)dealloc; {
#if NON_ARC
  [super dealloc]; // no-warning
#endif
}
@end


__attribute__((objc_root_class))
@interface NonNSObjectMissingDealloc
@property (retain) NSObject *ivar;
@end
@implementation NonNSObjectMissingDealloc
-(void)dealloc; {

}
@end

// Warn about calling -dealloc rather than release by mistake.

@interface CallDeallocOnRetainPropIvar : NSObject {
  NSObject *okToDeallocDirectly;
}

@property (retain) NSObject *ivar;
@end

@implementation CallDeallocOnRetainPropIvar
- (void)dealloc
{
#if NON_ARC
  // Only warn for synthesized ivars.
  [okToDeallocDirectly dealloc]; // no-warning
  [_ivar dealloc];  // expected-warning {{'_ivar' should be released rather than deallocated}}

  [super dealloc];
#endif
}
@end

// CIFilter special cases.
// By design, -[CIFilter dealloc] releases (by calling -setValue: forKey: with
// 'nil') all ivars (even in its *subclasses*) with names starting with
// 'input' or that are backed by properties with names starting with 'input'.
// The Dealloc checker needs to take particular care to not warn about missing
// releases in this case -- if the user adds a release quiet the
// warning it may result in an over release.

@interface ImmediateSubCIFilter : CIFilter {
  NSObject *inputIvar;
  NSObject *nonInputIvar;
  NSObject *notPrefixedButBackingPrefixedProperty;
  NSObject *inputPrefixedButBackingNonPrefixedProperty;
}

@property(retain) NSObject *inputIvar;
@property(retain) NSObject *nonInputIvar;
@property(retain) NSObject *inputAutoSynthesizedIvar;
@property(retain) NSObject *inputExplicitlySynthesizedToNonPrefixedIvar;
@property(retain) NSObject *nonPrefixedPropertyBackedByExplicitlySynthesizedPrefixedIvar;

@end

@implementation ImmediateSubCIFilter
@synthesize inputIvar = inputIvar;
@synthesize nonInputIvar = nonInputIvar;
@synthesize inputExplicitlySynthesizedToNonPrefixedIvar = notPrefixedButBackingPrefixedProperty;
@synthesize nonPrefixedPropertyBackedByExplicitlySynthesizedPrefixedIvar = inputPrefixedButBackingNonPrefixedProperty;

- (void)dealloc {
#if NON_ARC
  // We don't want warnings here for:
  // inputIvar
  // inputAutoSynthesizedIvar
  // inputExplicitlySynthesizedToNonPrefixedIvar
  // inputPrefixedButBackingNonPrefixedProperty
  [super dealloc];
  // expected-warning@-1 {{The 'nonInputIvar' ivar in 'ImmediateSubCIFilter' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}
@end

@interface SubSubCIFilter : CIFilter {
  NSObject *inputIvarInSub;
}

@property(retain) NSObject *inputIvarInSub;
@end

@implementation SubSubCIFilter
@synthesize inputIvarInSub = inputIvarInSub;

- (void)dealloc {
// Don't warn about inputIvarInSub.
#if NON_ARC
  [super dealloc];
#endif
}
@end
@interface OverreleasingCIFilter : CIFilter {
  NSObject *inputIvar;
}

@property(retain) NSObject *inputIvar;
@end

@implementation OverreleasingCIFilter
@synthesize inputIvar = inputIvar;

- (void)dealloc {
#if NON_ARC
  // This is an over release because CIFilter's dealloc will also release it.
  [inputIvar release]; // expected-warning {{The 'inputIvar' ivar in 'OverreleasingCIFilter' will be released by '-[CIFilter dealloc]' but also released here}}
  [super dealloc]; // no-warning
  #endif
}
@end


@interface NotMissingDeallocCIFilter : CIFilter {
  NSObject *inputIvar;
}

@property(retain) NSObject *inputIvar;
@end

@implementation NotMissingDeallocCIFilter // no-warning
@synthesize inputIvar = inputIvar;
@end


@interface ClassWithRetainPropWithIBOutletIvarButNoSetter : NSObject {
  // On macOS, the nib-loading code will set the ivar directly without
  // retaining value (unike iOS, where it is retained). This means that
  // on macOS we should not warn about a missing release for a property backed
  // by an IBOutlet ivar when that property does not have a setter.
  IBOutlet NSObject *ivarForOutlet;
}

@property (readonly, retain) NSObject *ivarForOutlet;
@end

@implementation ClassWithRetainPropWithIBOutletIvarButNoSetter

@synthesize ivarForOutlet;
- (void)dealloc {

#if NON_ARC
  [super dealloc];
#if !MACOS
// expected-warning@-2{{The 'ivarForOutlet' ivar in 'ClassWithRetainPropWithIBOutletIvarButNoSetter' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
#endif
}

@end

@interface ClassWithRetainPropWithIBOutletIvarAndShadowingReadWrite : NSObject {
  IBOutlet NSObject *ivarForOutlet;
}

@property (readonly, retain) NSObject *ivarForOutlet;

@end

@interface ClassWithRetainPropWithIBOutletIvarAndShadowingReadWrite ()

// Since there is a shadowing readwrite property, there will be a retaining
// setter and so the ivar will be retained by nib-loading code even on
// macOS and therefore must be released.
@property (readwrite, retain) NSObject *ivarForOutlet;
@end

@implementation ClassWithRetainPropWithIBOutletIvarAndShadowingReadWrite

@synthesize ivarForOutlet;
- (void)dealloc {

#if NON_ARC
  [super dealloc];
// expected-warning@-1{{The 'ivarForOutlet' ivar in 'ClassWithRetainPropWithIBOutletIvarAndShadowingReadWrite' was retained by a synthesized property but not released before '[super dealloc]'}}
#endif
}

@end
