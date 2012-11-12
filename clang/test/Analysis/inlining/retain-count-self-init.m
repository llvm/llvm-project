// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,osx.cocoa.SelfInit -analyzer-ipa=dynamic-bifurcate -verify %s

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
@end

// We do not want to overhelm user with error messages in case they forgot to 
// assign to self and check that the result of [super init] is non-nil. So 
// stop tracking the receiver of init with respect to Retain Release checker.  
// radar://12115830
@interface ParentOfCell : NSObject
- (id)initWithInt: (int)inInt;
@end
@interface Cell : ParentOfCell{
  int x;
}
- (id)init;
+ (void)test;
@property int x;
@end
@implementation Cell
@synthesize x;
- (id) init {
  [super init];
  self.x = 3; // no-warning 
  return self; // expected-warning {{Returning 'self' while it is not set to the result of '[(super or self)}} 
}
- (id) initWithInt: (int)inInt {
  [super initWithInt: inInt];
  self.x = inInt; // no-warning 
  return self; // expected-warning {{Returning 'self' while it is not set to the result of '[(super or self)}} 
}
- (id) init2 {
  [self init]; // The call [self init] is inlined. We will warn inside the inlined body.
  self.x = 2; // no-warning 
  return self; 
}

- (id) initWithIntGood: (int)inInt {
    if (self = [super initWithInt: inInt]) {
      self.x = inInt; 
    }
    return self; 
}
+ (void) test {
  Cell *sharedCell1 = [[Cell alloc] init];
  [sharedCell1 release];
  Cell *sharedCell2 = [[Cell alloc] initWithInt: 3];
  [sharedCell2 release];
  Cell *sharedCell3 = [[Cell alloc] initWithIntGood: 3];
  [sharedCell3 release];
}

@end

