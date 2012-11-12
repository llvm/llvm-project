// RUN: %clang_cc1  -fsyntax-only -fblocks -verify %s
// rdar://10111397

#if __LP64__
typedef unsigned long NSUInteger;
#else
typedef unsigned int NSUInteger;
#endif

@interface NSObject
+ (NSObject*)nsobject;
@end

@interface NSNumber : NSObject
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithFloat:(float)value;
@end

int main() {
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
}

// Dictionary test
@class NSDictionary;

NSDictionary *err() {
  return @{@"name" : @"value"}; // expected-error {{declaration of 'dictionaryWithObjects:forKeys:count:' is missing in NSDictionary class}}
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

id NSUserName();

int Int();

NSDictionary * blocks() {
  return @{ @"task" : ^ { return 17; } };
}

NSDictionary * warn() {
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

// rdar:// 11231426
typedef float BOOL;

BOOL radar11231426() {
        return __objc_yes;
}

id stringBoxingNoSuchMethod(const char *str) {
  return @(str); // expected-error {{declaration of 'stringWithUTF8String:' is missing in NSString class}}
}
