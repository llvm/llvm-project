// RUN: %clang_cc1  -fsyntax-only -fblocks -triple x86_64-apple-darwin10 -verify %s

#if __LP64__
typedef unsigned long NSUInteger;
typedef long NSInteger;
#else
typedef unsigned int NSUInteger;
typedef int NSInteger;
#endif

void checkNSNumberUnavailableDiagnostic(void) {
  id num = @1000; // expected-error {{definition of class NSNumber must be available to use Objective-C numeric literals}}

  int x = 1000;
  id num1 = @(x); // expected-error {{definition of class NSNumber must be available to use Objective-C numeric literals}}\
                  // expected-error {{illegal type 'int' used in a boxed expression}}
}

@class NSNumber; // expected-note 2 {{forward declaration of class here}}

void checkNSNumberFDDiagnostic(void) {
  id num = @1000; // expected-error {{definition of class NSNumber must be available to use Objective-C numeric literals}}

  int x = 1000;
  id num1 = @(x); // expected-error {{definition of class NSNumber must be available to use Objective-C numeric literals}}\
                  // expected-error {{illegal type 'int' used in a boxed expression}}
}

@interface NSObject
+ (NSObject*)nsobject;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithInteger:(NSInteger)value ;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value ;
@end

int big = 1391126400;
int thousand = 1000;
int main(void) {
  NSNumber * N = @3.1415926535;  // expected-error {{declaration of 'numberWithDouble:' is missing in NSNumber class}}
  NSNumber *noNumber = @__objc_yes; // expected-error {{declaration of 'numberWithBool:' is missing in NSNumber class}}
  NSNumber * NInt = @1000;
  NSNumber * NLongDouble = @1000.0l; // expected-error{{'long double' is not a valid literal type for NSNumber}}
  id character = @ 'a';

  NSNumber *NNegativeInt = @-1000;
  NSNumber *NPositiveInt = @+1000;
  NSNumber *NNegativeFloat = @-1000.1f;
  NSNumber *NPositiveFloat = @+1000.1f;

  int five = 5;
  @-five; // expected-error{{@- must be followed by a number to form an NSNumber object}}
  @+five; // expected-error{{@+ must be followed by a number to form an NSNumber object}}
  NSNumber *av = @(1391126400000);
  NSNumber *bv = @(1391126400 * 1000); // expected-warning {{overflow in expression; result is -443003904 with type 'int'}}
  NSNumber *cv = @(big * thousand);
}

// Dictionary test
@class NSDictionary;  // expected-note {{forward declaration of class here}}

NSDictionary *err(void) {
  return @{@"name" : @"value"}; // expected-error {{definition of class NSDictionary must be available to use Objective-C dictionary literals}}
}

@interface NSDate : NSObject
+ (NSDate *) date;
@end

@protocol NSCopying
- copy;
@end

@interface NSDictionary : NSObject
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id<NSCopying> [])keys count:(NSUInteger)cnt;
@end

@interface NSString<NSCopying>
@end

id NSUserName(void);

int Int(void);

NSDictionary * blocks(void) {
  return @{ @"task" : ^ { return 17; } };
}

NSDictionary * warn(void) {
  NSDictionary *dictionary = @{@"name" : NSUserName(),
                               @"date" : [NSDate date],
                               @"name2" : @"other",
                               NSObject.nsobject : @"nsobject" }; // expected-warning{{passing 'NSObject *' to parameter of incompatible type 'const id<NSCopying>'}}
  NSDictionary *dictionary2 = @{@"name" : Int()}; // expected-error {{collection element of type 'int' is not an Objective-C object}}

  NSObject *o;
  NSDictionary *dictionary3 = @{o : o, // expected-warning{{passing 'NSObject *' to parameter of incompatible type 'const id<NSCopying>'}}
                               @"date" : [NSDate date] };
  return dictionary3;
}

typedef float BOOL;

BOOL radar11231426(void) {
        return __objc_yes;
}

id stringBoxingNoSuchMethod(const char *str) {
  return @(str); // expected-error {{declaration of 'stringWithUTF8String:' is missing in NSString class}}
}
