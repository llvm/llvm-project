// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s 
// rdar://10803676

extern "C" void *sel_registerName(const char *);

typedef bool BOOL;
typedef long NSInteger;
typedef unsigned long NSUInteger;

#if __has_feature(objc_bool)
#define YES             __objc_yes
#define NO              __objc_no
#else
#define YES             ((BOOL)1)
#define NO              ((BOOL)0)
#endif

@interface NSNumber
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
+ (NSNumber *)numberWithInteger:(NSInteger)value ;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value ;
@end

int main(int argc, const char *argv[]) {
  // character literals.
  NSNumber *theLetterZ = @'Z';          // equivalent to [NSNumber numberWithChar:'Z']

  // integral literals.
  NSNumber *fortyTwo = @42;             // equivalent to [NSNumber numberWithInt:42]
  NSNumber *fortyTwoUnsigned = @42U;    // equivalent to [NSNumber numberWithUnsignedInt:42U]
  NSNumber *fortyTwoLong = @42L;        // equivalent to [NSNumber numberWithLong:42L]
  NSNumber *fortyTwoLongLong = @42LL;   // equivalent to [NSNumber numberWithLongLong:42LL]

  // floating point literals.
  NSNumber *piFloat = @3.141592654F;    // equivalent to [NSNumber numberWithFloat:3.141592654F]
  NSNumber *piDouble = @3.1415926535;   // equivalent to [NSNumber numberWithDouble:3.1415926535]

  // BOOL literals.
  NSNumber *yesNumber = @YES;           // equivalent to [NSNumber numberWithBool:YES]
  NSNumber *noNumber = @NO;             // equivalent to [NSNumber numberWithBool:NO]

  NSNumber *trueNumber = @true;         // equivalent to [NSNumber numberWithBool:(BOOL)true]
  NSNumber *falseNumber = @false;       // equivalent to [NSNumber numberWithBool:(BOOL)false]
}

// CHECK:  NSNumber *theLetterZ = ((NSNumber *(*)(id, SEL, char))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithChar:"), 'Z');
// CHECK:  NSNumber *fortyTwo = ((NSNumber *(*)(id, SEL, int))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithInt:"), 42);
// CHECK:  NSNumber *fortyTwoUnsigned = ((NSNumber *(*)(id, SEL, unsigned int))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithUnsignedInt:"), 42U);
// CHECK:  NSNumber *fortyTwoLong = ((NSNumber *(*)(id, SEL, long))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithLong:"), 42L);
// CHECK:  NSNumber *fortyTwoLongLong = ((NSNumber *(*)(id, SEL, long long))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithLongLong:"), 42LL);
// CHECK:  NSNumber *piFloat = ((NSNumber *(*)(id, SEL, float))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithFloat:"), 3.1415927F);
// CHECK:  NSNumber *piDouble = ((NSNumber *(*)(id, SEL, double))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithDouble:"), 3.1415926535);
// CHECK:  NSNumber *yesNumber = ((NSNumber *(*)(id, SEL, BOOL))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithBool:"), true);
// CHECK:  NSNumber *noNumber = ((NSNumber *(*)(id, SEL, BOOL))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithBool:"), false);
// CHECK:  NSNumber *trueNumber = ((NSNumber *(*)(id, SEL, BOOL))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithBool:"), true);
// CHECK:  NSNumber *falseNumber = ((NSNumber *(*)(id, SEL, BOOL))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithBool:"), false);
