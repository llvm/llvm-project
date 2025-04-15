// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

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

// CHECK: [[STRUCT_NSCONSTANT_STRING_TAG:%.*]] = type { ptr, i32, ptr, i64 }

// CHECK: [[WithIntMeth:@.*]] = private unnamed_addr constant [15 x i8] c"numberWithInt:\00"
// CHECK: [[WithIntSEL:@.*]] = internal externally_initialized global ptr [[WithIntMeth]]
// CHECK: [[WithCharMeth:@.*]] = private unnamed_addr constant [16 x i8] c"numberWithChar:\00"
// CHECK: [[WithCharSEL:@.*]] = internal externally_initialized global ptr [[WithCharMeth]]
// CHECK: [[WithBoolMeth:@.*]] = private unnamed_addr constant [16 x i8] c"numberWithBool:\00"
// CHECK: [[WithBoolSEL:@.*]] = internal externally_initialized global ptr [[WithBoolMeth]]
// CHECK: [[WithIntegerMeth:@.*]] = private unnamed_addr constant [19 x i8] c"numberWithInteger:\00"
// CHECK: [[WithIntegerSEL:@.*]] = internal externally_initialized global ptr [[WithIntegerMeth]]
// CHECK: [[WithUnsignedIntegerMeth:@.*]] = private unnamed_addr constant [27 x i8] c"numberWithUnsignedInteger:\00"
// CHECK: [[WithUnsignedIntegerSEL:@.*]] = internal externally_initialized global ptr [[WithUnsignedIntegerMeth]]
// CHECK: [[stringWithUTF8StringMeth:@.*]] = private unnamed_addr constant [22 x i8] c"stringWithUTF8String:\00"
// CHECK: [[stringWithUTF8StringSEL:@.*]] = internal externally_initialized global ptr [[stringWithUTF8StringMeth]]
// CHECK: [[STR0:.*]] = private unnamed_addr constant [4 x i8] c"abc\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: [[UNNAMED_CFSTRING:.*]] = private global [[STRUCT_NSCONSTANT_STRING_TAG]] { ptr @__CFConstantStringClassReference, i32 1992, ptr [[STR0]], i64 3 }, section "__DATA,__cfstring", align 8

int main(void) {
  // CHECK: [[T:%t]] = alloca ptr, align 8

  // CHECK: load ptr, ptr [[WithIntSEL]]
  int i; @(i);
  // CHECK: load ptr, ptr [[WithCharSEL]]
  signed char sc; @(sc);
  // CHECK: load ptr, ptr [[WithBoolSEL]]
  BOOL b; @(b);
  // CHECK: load ptr, ptr [[WithBoolSEL]]
  typeof(b) b2; @(b2);
  // CHECK: load ptr, ptr [[WithBoolSEL]]
  typedef const typeof(b) MyBOOL; MyBOOL b3; @(b3);
  // CHECK: load ptr, ptr [[WithBoolSEL]]
  @((BOOL)i);
  // CHECK: load ptr, ptr [[WithIntegerSEL]]
  @((NSInteger)i);
  // CHECK: load ptr, ptr [[WithUnsignedIntegerSEL]]
  @((NSUInteger)i);
  // CHECK: load ptr, ptr [[stringWithUTF8StringSEL]]
  const char *s; @(s);

  typedef enum : NSInteger { Red, Green, Blue } Color;
  // CHECK: load ptr, ptr [[WithIntegerSEL]]
  @(Red);
  Color col = Red;
  // CHECK: load ptr, ptr [[WithIntegerSEL]]
  @(col);

  // CHECK: store ptr [[UNNAMED_CFSTRING]], ptr [[T]], align 8
  NSString *t = @("abc");
}
