// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-config ipa=dynamic-bifurcate -verify %s

typedef signed char BOOL;
typedef struct objc_class *Class;
typedef struct objc_object {
    Class isa;
} *id;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
+(id)new;
- (oneway void)release;
-(id)init;
-(id)autorelease;
-(id)copy;
- (Class)class;
-(id)retain;
- (oneway void)release;
@end

@interface SelfStaysLive : NSObject
- (id)init;
@end

@implementation SelfStaysLive
- (id)init {
  return [super init];
}
@end

void selfStaysLive(void) {
    SelfStaysLive *foo = [[SelfStaysLive alloc] init]; 
    [foo release];
}

// Test that retain release checker warns on leaks and use-after-frees when 
// self init is not enabled.  
@interface ParentOfCell : NSObject
- (id)initWithInt: (int)inInt;
@end
@interface Cell : ParentOfCell{
  int x;
}
- (id)initWithInt: (int)inInt;
+ (void)testOverRelease;
+ (void)testLeak;
@property int x;
@end
@implementation Cell
@synthesize x;
- (id) initWithInt: (int)inInt {
  [super initWithInt: inInt];
  self.x = inInt; // no-warning 
  return self; // Self Init checker would produce a warning here.
}
+ (void) testOverRelease {
  Cell *sharedCell3 = [[Cell alloc] initWithInt: 3];
  [sharedCell3 release];
  [sharedCell3 release]; // expected-warning {{Reference-counted object is used after it is released}}
}
+ (void) testLeak {
  Cell *sharedCell4 = [[Cell alloc] initWithInt: 3]; // expected-warning {{leak}}
}
@end
  
// We should stop tracking some objects even when we inline the call. 
// Specialically, the objects passed into calls with delegate and callback 
// parameters.
@class DelegateTest;
typedef void (*ReleaseCallbackTy) (DelegateTest *c);

@interface Delegate : NSObject
@end

@interface DelegateTest : NSObject {
  Delegate *myDel;
}
// Object initialized with a delagate which could potentially release it.
- (id)initWithDelegate: (id) d;

- (void) setDelegate: (id) d;

// Releases object through callback.
+ (void)updateObject:(DelegateTest*)obj WithCallback:(ReleaseCallbackTy)rc;

+ (void)test: (Delegate *)d;

@property (assign) Delegate* myDel;
@end

void releaseObj(DelegateTest *c);

// Releases object through callback.
void updateObject(DelegateTest *c, ReleaseCallbackTy rel) {
  rel(c);
}

@implementation DelegateTest
@synthesize myDel;

- (id) initWithDelegate: (id) d {
    if ((self = [super init]))
      myDel = d;
    return self;
}

- (void) setDelegate: (id) d {
    myDel = d;
}

+ (void)updateObject:(DelegateTest*)obj WithCallback:(ReleaseCallbackTy)rc {
  rc(obj);
}

+ (void) test: (Delegate *)d {
  DelegateTest *obj1 = [[DelegateTest alloc] initWithDelegate: d]; // no-warning
  DelegateTest *obj2 = [[DelegateTest alloc] init]; // no-warning
  DelegateTest *obj3 = [[DelegateTest alloc] init]; // no-warning
  updateObject(obj2, releaseObj);
  [DelegateTest updateObject: obj3
        WithCallback: releaseObj];
  DelegateTest *obj4 = [[DelegateTest alloc] init]; // no-warning
  [obj4 setDelegate: d];
}
@end

