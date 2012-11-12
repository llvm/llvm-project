// RUN: %clang_cc1 -analyze -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -verify -Wno-objc-root-class %s

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
  
  return [value autorelease]; // expected-warning {{Object sent -autorelease too many times}}
}

@end

NSNumber* numberFromMyNumberProperty(MyNumber* aMyNumber)
{
  NSNumber* result = aMyNumber.myNumber;
    
  return [result autorelease]; // expected-warning {{Object sent -autorelease too many times}}
}


// rdar://6611873

@interface Person : NSObject {
  NSString *_name;
}
@property (retain) NSString * name;
@end

@implementation Person
@synthesize name = _name;
@end

void rdar6611873() {
  Person *p = [[[Person alloc] init] autorelease];
  
  p.name = [[NSString string] retain]; // expected-warning {{leak}}
  p.name = [[NSString alloc] init]; // expected-warning {{leak}}
}

@interface SubPerson : Person
-(NSString *)foo;
@end

@implementation SubPerson
-(NSString *)foo {
  return super.name;
}
@end

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


