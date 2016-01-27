// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.Dealloc -fblocks %s 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.Dealloc -fblocks -triple x86_64-apple-darwin10 -fobjc-arc -fobjc-runtime-has-weak %s 2>&1 | FileCheck -check-prefix=CHECK-ARC -allow-empty '--implicit-check-not=error:' '--implicit-check-not=warning:' %s

#define nil ((id)0)

#define NON_ARC !__has_feature(objc_arc)

#if NON_ARC
#define WEAK_ON_ARC
#else
#define WEAK_ON_ARC __weak
#endif

typedef signed char BOOL;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (Class)class;
@end

@interface NSObject <NSObject> {}
- (void)dealloc;
- (id)init;
- (id)retain;
- (oneway void)release;
@end

typedef struct objc_selector *SEL;

//===------------------------------------------------------------------------===
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

//===------------------------------------------------------------------------===
// Warn about missing release in -dealloc for properties.

@interface MyPropertyClass1 : NSObject
// CHECK: DeallocMissingRelease.m:[[@LINE+1]]:1: warning: The '_ivar' instance variable in 'MyPropertyClass1' was retained by a synthesized property but was not released in 'dealloc'
@property (copy) NSObject *ivar;
@end

@implementation MyPropertyClass1
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface MyPropertyClass2 : NSObject
// CHECK: DeallocMissingRelease.m:[[@LINE+1]]:1: warning: The '_ivar' instance variable in 'MyPropertyClass2' was retained by a synthesized property but was not released in 'dealloc'
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClass2
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface MyPropertyClass3 : NSObject {
  NSObject *_ivar;
}
@property (retain) NSObject *ivar;
@end

@implementation MyPropertyClass3
// CHECK: DeallocMissingRelease.m:[[@LINE+1]]:1: warning: The '_ivar' instance variable in 'MyPropertyClass3' was retained by a synthesized property but was not released in 'dealloc'
@synthesize ivar = _ivar;
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface MyPropertyClass4 : NSObject {
  void (^_blockPropertyIvar)(void);
}
@property (copy) void (^blockProperty)(void);
@end

@implementation MyPropertyClass4
// CHECK: DeallocMissingRelease.m:[[@LINE+1]]:1: warning: The '_blockPropertyIvar' instance variable in 'MyPropertyClass4' was retained by a synthesized property but was not released in 'dealloc'
@synthesize blockProperty = _blockPropertyIvar;
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

@interface MyPropertyClass5 : NSObject {
  WEAK_ON_ARC NSObject *_ivar;
}
@property (weak) NSObject *ivar;
@end

@implementation MyPropertyClass5
@synthesize ivar = _ivar; // no-warning
- (void)dealloc
{
#if NON_ARC
  [super dealloc];
#endif
}
@end

//===------------------------------------------------------------------------===
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
// CHECK: 4 warnings generated.
