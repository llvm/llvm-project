// UNSUPPORTED: -zos, -aix
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -verify %s
// RUN: %clang_cc1 -include-pch %t -ast-print %s | FileCheck -check-prefix=CHECK-PRINT %s
// RUN: %clang_cc1 -include-pch %t -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-IR %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef unsigned char BOOL;

@interface NSNumber @end

@interface NSNumber (NSNumberCreation)
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
@end

@interface NSArray
@end

@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

// CHECK-IR: define internal {{.*}}void @test_numeric_literals()
static inline void test_numeric_literals(void) {
  // CHECK-PRINT: id intlit = @17
  // CHECK-IR: {{call.*17}}
  id intlit = @17;
  // CHECK-PRINT: id floatlit = @17.449999999999999
  // CHECK-IR: {{call.*1.745}}
  id floatlit = @17.45;
}

static inline void test_array_literals(void) {
  // CHECK-PRINT: id arraylit = @[ @17, @17.449999999999999
  id arraylit = @[@17, @17.45];
}

static inline void test_dictionary_literals(void) {
  // CHECK-PRINT: id dictlit = @{ @17 : {{@17.449999999999999[^,]*}}, @"hello" : @"world" };
  id dictlit = @{@17 : @17.45, @"hello" : @"world" };
}

#else
void test_all(void) {
  test_numeric_literals();
  test_array_literals();
  test_dictionary_literals();
}
#endif
