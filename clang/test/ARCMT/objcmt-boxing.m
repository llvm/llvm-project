// RUN: rm -rf %t
// RUN: %clang_cc1 -fobjc-arc -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c++ -verify
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c++ %s.result

#define YES __objc_yes
#define NO __objc_no

typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef signed char BOOL;
#define nil ((void*) 0)

#define INT_MIN   (-__INT_MAX__  -1)

@interface NSObject
+ (id)alloc;
@end

@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
- (id)initWithChar:(char)value;
- (id)initWithUnsignedChar:(unsigned char)value;
- (id)initWithShort:(short)value;
- (id)initWithUnsignedShort:(unsigned short)value;
- (id)initWithInt:(int)value;
- (id)initWithUnsignedInt:(unsigned int)value;
- (id)initWithLong:(long)value;
- (id)initWithUnsignedLong:(unsigned long)value;
- (id)initWithLongLong:(long long)value;
- (id)initWithUnsignedLongLong:(unsigned long long)value;
- (id)initWithFloat:(float)value;
- (id)initWithDouble:(double)value;
- (id)initWithBool:(BOOL)value;
- (id)initWithInteger:(NSInteger)value;
- (id)initWithUnsignedInteger:(NSUInteger)value;

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
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
+ (NSNumber *)numberWithInteger:(NSInteger)value;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value;
@end

enum {
    NSASCIIStringEncoding = 1,
    NSUTF8StringEncoding = 4,
    NSUnicodeStringEncoding = 10
};
typedef NSUInteger NSStringEncoding;

@interface NSString : NSObject
@end

@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
+ (id)stringWithCString:(const char *)cString encoding:(NSStringEncoding)enc;
+ (id)stringWithCString:(const char *)bytes;
- (instancetype)initWithUTF8String:(const char *)nullTerminatedCString;
@end

enum MyEnm {
  ME_foo
};

void foo() {
  [NSNumber numberWithInt:INT_MIN];
  bool cppb;
  [NSNumber numberWithBool:cppb];
  MyEnm myenum; 
  [NSNumber numberWithInteger:myenum];
  [NSNumber numberWithInteger:ME_foo];
  [NSNumber numberWithDouble:cppb]; // expected-warning {{converting to boxing syntax requires casting 'bool' to 'double'}}
}

void boxString() {
  NSString *s = [NSString stringWithUTF8String:"box"];
  const char *cstr1;
  char *cstr2;
  s = [NSString stringWithUTF8String:cstr1];
  s = [NSString stringWithUTF8String:cstr2];
  s = [NSString stringWithCString:cstr1 encoding:NSASCIIStringEncoding];
  s = [NSString stringWithCString:cstr1 encoding:NSUTF8StringEncoding];
  s = [NSString stringWithCString:cstr1 encoding: NSUnicodeStringEncoding];
  NSStringEncoding encode;
  s = [NSString stringWithCString:cstr1 encoding:encode];
  s = [NSString stringWithCString:cstr1];

  static const char strarr[] = "coolbox";
  s = [NSString stringWithUTF8String:strarr];
  // rdar://18080352
  const char *utf8Bytes = "blah";
  NSString *string1 = [NSString stringWithUTF8String:utf8Bytes];
  NSString *string2 = [[NSString alloc] initWithUTF8String:utf8Bytes];
}
