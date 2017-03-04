// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.SelfInit -analyzer-config ipa=dynamic -fno-builtin %s -verify
// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.SelfInit -fno-builtin %s -verify

@class NSZone, NSCoder;
@protocol NSObject
- (id)self;
@end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end 
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end 
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)allocWithZone:(NSZone *)zone;
+ (id)alloc;
- (void)dealloc;
-(id)class;
-(id)init;
-(id)release;
@end
@interface NSProxy <NSObject> {}
@end

//#import "Foundation/NSObject.h"
typedef unsigned NSUInteger;
typedef long NSInteger;

@interface NSInvocation : NSObject {}
- (void)getArgument:(void *)argumentLocation atIndex:(NSInteger)idx;
- (void)setArgument:(void *)argumentLocation atIndex:(NSInteger)idx;
@end

@class NSMethodSignature, NSCoder, NSString, NSEnumerator;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;

@interface NSBundle : NSObject
+(id)loadNibNamed:(NSString*)s owner:(id)o;
@end

void log(void *obj);
extern void *somePtr;

@class MyObj;
extern id _commonInit(MyObj *self);

@interface MyObj : NSObject {
	id myivar;
	int myint;
}
-(id)_init;
-(id)initWithSomething:(int)x;
-(void)doSomething;
+(id)commonInitMember:(id)s;
@end

@interface MyProxyObj : NSProxy {}
-(id)init;
@end

@implementation MyObj

-(id)init {
  do { if (!((somePtr != 0))) { [[NSAssertionHandler currentHandler] handleFailureInMethod:_cmd object:self file:[NSString stringWithUTF8String:"init.m"] lineNumber:21 description:(@"Invalid parameter not satisfying: %s"), ("x != 0"), (0), (0), (0), (0)]; } } while(0);
  return [self initWithSomething:0];
}

-(id)init2 {
  self = [self initWithSomething:0];
  return self;
}

-(id)init3 {
	log([self class]);
	return [self initWithSomething:0];
}

-(id)init4 {
	self = [super init];
	if (self) {
		log(&self);
	}
	return self;
}

-(id)init4_w {
  [super init];
  if (self) {
    log(&self);
  }
  return self; // expected-warning {{Returning 'self' while it is not set to the result of '[(super or self) init...]'}}
}

- (id)initWithSomething:(int)x {    
	if ((self = [super init]))
		myint = x;
	return self;
}

-(id)_init {
	myivar = 0;
	return self;
}

-(id)init5 {
  [NSBundle loadNibNamed:@"Window" owner:self];
  return [self initWithSomething:0];
}

-(id)init6 {
  [NSBundle loadNibNamed:@"Window" owner:myivar]; // no-warning
  return [self initWithSomething:0];
}

-(id)init7 {
  if (0 != (self = [self _init]))
    myivar = 0;
  return self;
}

-(id)init8 {
    if ((self = [super init])) {
		log(&self);
		myivar = 0;
    }
    return self;
}

-(id)init9 {
  [self doSomething];
  return self; // no-warning
}

-(id)init10 {
  myivar = 0; // no-warning
  return self;
}

-(id)init11 {
  return self; // no-warning
}

-(id)init12 {
	[super init];
	return self; // expected-warning {{Returning 'self'}}
}

-(id)init13 {
	if (self == [super init]) {
	  myivar = 0; // expected-warning {{Instance variable used}}
	}
	return self; // expected-warning {{Returning 'self'}}
}

-(id)init14 {
  if (!(self = _commonInit(self)))
    return 0;
  return self;
}

-(id)init14_w {
  [super init];
  self = _commonInit(self);
  return self; // expected-warning {{Returning 'self' while it is not set to the result of '[(super or self) init...]'}}
}

-(id)init15 {
  if (!(self = [super init]))
    return 0;
  return self;
}

-(id)init16 {
  somePtr = [super init];
  self = somePtr;
  myivar = 0; 
  return self;
}

-(id)init17 {
  somePtr = [super init];
  myivar = 0; // expected-warning {{Instance variable used}}
  return 0;
}

-(id)init18 {
  self = [super init];
  self = _commonInit(self);
  return self;
}

+(id)commonInitMember:(id)s {
  return s;
}

-(id)init19 {
  self = [super init];
  self = [MyObj commonInitMember:self];
  return self;
}

-(id)init19_w {
  [super init];
  self = [MyObj commonInitMember:self];
  return self; // expected-warning {{Returning 'self'}}
}

-(void)doSomething {}

@end

@implementation MyProxyObj

- (id)init { return self; }

@end


// Test for radar://10973514 : self should not be invalidated by a method call.
@interface Test : NSObject {
    NSInvocation *invocation_;
}
@end
@implementation Test
-(id) initWithTarget:(id) rec selector:(SEL) cb {
  if (self=[super init]) {
    [invocation_ setArgument:&self atIndex:2];
  }   
  return self;
}
@end

// Test radar:11235991 - passing self to a call to super.
@protocol MyDelegate
@end
@interface Object : NSObject
- (id) initWithObject: (id)i;
@end
@interface Derived: Object <MyDelegate>
- (id) initWithInt: (int)t;
@property (nonatomic, retain, readwrite) Object *size;
@end
@implementation Derived 
- (id) initWithInt: (int)t {
   if ((self = [super initWithObject:self])) {
      _size = [[Object alloc] init];
   }
   return self;
}
@end

// Test for radar://11125870: init constructing a special instance.
typedef signed char BOOL;
@interface MyClass : NSObject
@end
@implementation MyClass
+ (id)specialInstance {
    return [[MyClass alloc] init];
}
- (id)initSpecially:(BOOL)handleSpecially {
    if ((self = [super init])) {
        if (handleSpecially) {
            self = [MyClass specialInstance];
        }
    }
    return self;
}
- (id)initSelfSelf {
    if ((self = [super init])) {
      self = self;
    }
    return self;
}
@end

// Test for radar://12838705.
@interface ABCClass : NSObject
@property (nonatomic, strong) NSString *foo;
@property (nonatomic, strong) NSString *bar;
@property (nonatomic, strong) NSString *baz;
@end

@implementation ABCClass
@synthesize foo = foo_;
@synthesize bar = bar_;
@synthesize baz = baz_;

- (id)initWithABC:(ABCClass *)abc {
  self = [super init];
  baz_ = abc->baz_;
  return self;
}

- (ABCClass *)abcWithFoo:(NSString *)foo {
  ABCClass *copy = [[ABCClass alloc] initWithABC:self];
  return copy;
}

@end

