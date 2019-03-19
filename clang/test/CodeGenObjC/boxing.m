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

// CHECK: [[V0:%.*]] = type opaque
// CHECK: [[STRUCT_NSCONSTANT_STRING_TAG:%.*]] = type { i32*, i32, i8*, i64 }

// CHECK: [[WithIntMeth:@.*]] = private unnamed_addr constant [15 x i8] c"numberWithInt:\00"
// CHECK: [[WithIntSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[WithIntMeth]]
// CHECK: [[WithCharMeth:@.*]] = private unnamed_addr constant [16 x i8] c"numberWithChar:\00"
// CHECK: [[WithCharSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([16 x i8], [16 x i8]* [[WithCharMeth]]
// CHECK: [[WithBoolMeth:@.*]] = private unnamed_addr constant [16 x i8] c"numberWithBool:\00"
// CHECK: [[WithBoolSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([16 x i8], [16 x i8]* [[WithBoolMeth]]
// CHECK: [[WithIntegerMeth:@.*]] = private unnamed_addr constant [19 x i8] c"numberWithInteger:\00"
// CHECK: [[WithIntegerSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[WithIntegerMeth]]
// CHECK: [[WithUnsignedIntegerMeth:@.*]] = private unnamed_addr constant [27 x i8] c"numberWithUnsignedInteger:\00"
// CHECK: [[WithUnsignedIntegerSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([27 x i8], [27 x i8]* [[WithUnsignedIntegerMeth]]
// CHECK: [[stringWithUTF8StringMeth:@.*]] = private unnamed_addr constant [22 x i8] c"stringWithUTF8String:\00"
// CHECK: [[stringWithUTF8StringSEL:@.*]] = private externally_initialized global i8* getelementptr inbounds ([22 x i8], [22 x i8]* [[stringWithUTF8StringMeth]]
// CHECK: [[STR0:.*]] = private unnamed_addr constant [4 x i8] c"abc\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: [[UNNAMED_CFSTRING:.*]] = private global [[STRUCT_NSCONSTANT_STRING_TAG]] { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[STR0]], i32 0, i32 0), i64 3 }, section "__DATA,__cfstring", align 8

int main() {
  // CHECK: [[T:%.*]] = alloca [[V0]]*, align 8

  // CHECK: load i8*, i8** [[WithIntSEL]]
  int i; @(i);
  // CHECK: load i8*, i8** [[WithCharSEL]]
  signed char sc; @(sc);
  // CHECK: load i8*, i8** [[WithBoolSEL]]
  BOOL b; @(b);
  // CHECK: load i8*, i8** [[WithBoolSEL]]
  typeof(b) b2; @(b2);
  // CHECK: load i8*, i8** [[WithBoolSEL]]
  typedef const typeof(b) MyBOOL; MyBOOL b3; @(b3);
  // CHECK: load i8*, i8** [[WithBoolSEL]]
  @((BOOL)i);
  // CHECK: load i8*, i8** [[WithIntegerSEL]]
  @((NSInteger)i);
  // CHECK: load i8*, i8** [[WithUnsignedIntegerSEL]]
  @((NSUInteger)i);
  // CHECK: load i8*, i8** [[stringWithUTF8StringSEL]]
  const char *s; @(s);

  typedef enum : NSInteger { Red, Green, Blue } Color;
  // CHECK: load i8*, i8** [[WithIntegerSEL]]
  @(Red);
  Color col = Red;
  // CHECK: load i8*, i8** [[WithIntegerSEL]]
  @(col);

  // CHECK: store [[V0]]* bitcast ([[STRUCT_NSCONSTANT_STRING_TAG]]* [[UNNAMED_CFSTRING]] to [[V0]]*), [[V0]]** [[T]], align 8
  NSString *t = @("abc");
}
