// RUN: %clang_cc1 -analyze -analyze-function="-[Test1 myMethodWithY:withX:]" -analyzer-checker=core,osx.cocoa.RetainCount -analyzer-store=region -verify %s

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

@interface Test1 : NSObject {
  NSString *text;
}
-(id)myMethod;
-(id)myMethodWithY:(int)Y withX:(int)X;

@property (nonatomic, assign) NSString *text;
@end

@implementation Test1

@synthesize text;

-(id)myMethod {
  Test1 *cell = [[[Test1 alloc] init] autorelease];

  NSString *string1 = [[NSString alloc] initWithFormat:@"test %f", 0.0]; // No warning: this function is not analized.
  cell.text = string1;

  return cell;
}

-(id)myMethodWithY:(int)Y withX:(int)X {
  Test1 *cell = [[[Test1 alloc] init] autorelease];

  NSString *string1 = [[NSString alloc] initWithFormat:@"test %f %d", 0.0, X+Y]; // expected-warning {{Potential leak}}
  cell.text = string1;

  return cell;
}

@end
