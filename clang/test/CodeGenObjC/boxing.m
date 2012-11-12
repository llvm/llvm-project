// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef signed char BOOL;
#define nil ((void*) 0)

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

@interface NSString : NSObject
@end

@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

// CHECK: [[WithIntMeth:@".*"]] = internal global [15 x i8] c"numberWithInt:\00"
// CHECK: [[WithIntSEL:@".*"]] = internal global i8* getelementptr inbounds ([15 x i8]* [[WithIntMeth]]
// CHECK: [[WithCharMeth:@".*"]] = internal global [16 x i8] c"numberWithChar:\00"
// CHECK: [[WithCharSEL:@".*"]] = internal global i8* getelementptr inbounds ([16 x i8]* [[WithCharMeth]]
// CHECK: [[WithBoolMeth:@".*"]] = internal global [16 x i8] c"numberWithBool:\00"
// CHECK: [[WithBoolSEL:@".*"]] = internal global i8* getelementptr inbounds ([16 x i8]* [[WithBoolMeth]]
// CHECK: [[WithIntegerMeth:@".*"]] = internal global [19 x i8] c"numberWithInteger:\00"
// CHECK: [[WithIntegerSEL:@".*"]] = internal global i8* getelementptr inbounds ([19 x i8]* [[WithIntegerMeth]]
// CHECK: [[WithUnsignedIntegerMeth:@".*"]] = internal global [27 x i8] c"numberWithUnsignedInteger:\00"
// CHECK: [[WithUnsignedIntegerSEL:@".*"]] = internal global i8* getelementptr inbounds ([27 x i8]* [[WithUnsignedIntegerMeth]]
// CHECK: [[stringWithUTF8StringMeth:@".*"]] = internal global [22 x i8] c"stringWithUTF8String:\00"
// CHECK: [[stringWithUTF8StringSEL:@".*"]] = internal global i8* getelementptr inbounds ([22 x i8]* [[stringWithUTF8StringMeth]]

int main() {
  // CHECK: load i8** [[WithIntSEL]]
  int i; @(i);
  // CHECK: load i8** [[WithCharSEL]]
  signed char sc; @(sc);
  // CHECK: load i8** [[WithBoolSEL]]
  BOOL b; @(b);
  // CHECK: load i8** [[WithBoolSEL]]
  typeof(b) b2; @(b2);
  // CHECK: load i8** [[WithBoolSEL]]
  typedef const typeof(b) MyBOOL; MyBOOL b3; @(b3);
  // CHECK: load i8** [[WithBoolSEL]]
  @((BOOL)i);
  // CHECK: load i8** [[WithIntegerSEL]]
  @((NSInteger)i);
  // CHECK: load i8** [[WithUnsignedIntegerSEL]]
  @((NSUInteger)i);
  // CHECK: load i8** [[stringWithUTF8StringSEL]]
  const char *s; @(s);

  typedef enum : NSInteger { Red, Green, Blue } Color;
  // CHECK: load i8** [[WithIntegerSEL]]
  @(Red);
  Color col = Red;
  // CHECK: load i8** [[WithIntegerSEL]]
  @(col);
}
