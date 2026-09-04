// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -fobjc-constant-literals -fconstant-nsnumber-literals -fconstant-nsarray-literals -fconstant-nsdictionary-literals -include-pch %t -verify %s

// expected-no-diagnostics

#if __has_feature(objc_constant_literals)

#ifndef HEADER
#define HEADER

typedef unsigned char BOOL;

@interface NSNumber
@end

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
+ (id)arrayWithObjects:(const id[])objects count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id[])objects forKeys:(const id[])keys count:(unsigned long)cnt;
@end

static NSNumber *const intlit = @17;
static NSNumber *const floatlit = @17.45f;
static NSNumber *const doublelit = @17.45;

static NSDictionary *const dictlit = @{@"hello" : @17,
                                       @"world" : @17.45};
static NSDictionary *const dictlitIdentical = @{@"hello" : @17,
                                                @"world" : @17.45};

static NSArray *const arraylit = @[ @17, @17.45, @17.45f ];
static NSArray *const arraylitIdentical = @[ @17, @17.45, @17.45f ];
static NSArray *const arrayWithDictKeys = @[ @"hello", @"world" ];

#else
void test_all() {
  NSNumber *a = intlit;
  NSNumber *b = floatlit;
  NSNumber *c = doublelit;
  NSDictionary *d = dictlit;
  NSDictionary *e = dictlitIdentical;
  NSArray *f = arraylit;
  NSArray *g = arraylitIdentical;
  NSArray *h = arrayWithDictKeys;
}
#endif /* ! HEADER */

#else
#error
#endif /* ! __has_feature(objc_constant_literals) */