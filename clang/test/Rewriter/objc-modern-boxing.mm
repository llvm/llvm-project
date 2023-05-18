// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Werror -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp -Wno-attributes

extern char *strdup(const char *str);
extern "C" void *sel_registerName(const char *);

typedef signed char BOOL;
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

@interface NSString
+ (id)stringWithUTF8String:(const char *)str;
@end

int main(int argc, const char *argv[]) {
  // character.
  NSNumber *theLetterZ = @('Z');          // equivalent to [NSNumber numberWithChar:('Z')]

  // integral.
  NSNumber *fortyTwo = @(42);             // equivalent to [NSNumber numberWithInt:(42)]
  NSNumber *fortyTwoUnsigned = @(42U);    // equivalent to [NSNumber numberWithUnsignedInt:(42U)]
  NSNumber *fortyTwoLong = @(42L);        // equivalent to [NSNumber numberWithLong:(42L)]
  NSNumber *fortyTwoLongLong = @(42LL);   // equivalent to [NSNumber numberWithLongLong:(42LL)]

  // floating point.
  NSNumber *piFloat = @(3.141592654F);    // equivalent to [NSNumber numberWithFloat:(3.141592654F)]
  NSNumber *piDouble = @(3.1415926535);   // equivalent to [NSNumber numberWithDouble:(3.1415926535)]

  BOOL b;
  NSNumber *nsb = @(b);

  // Strings.
  NSString *duplicateString = @(strdup("Hello"));
}

// CHECK:  NSNumber *theLetterZ = ((NSNumber *(*)(Class, SEL, char))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithChar:"), ('Z'));
// CHECK:  NSNumber *fortyTwo = ((NSNumber *(*)(Class, SEL, int))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithInt:"), (42));
// CHECK:  NSNumber *fortyTwoUnsigned = ((NSNumber *(*)(Class, SEL, unsigned int))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithUnsignedInt:"), (42U));
// CHECK:  NSNumber *fortyTwoLong = ((NSNumber *(*)(Class, SEL, long))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithLong:"), (42L));
// CHECK:  NSNumber *fortyTwoLongLong = ((NSNumber *(*)(Class, SEL, long long))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithLongLong:"), (42LL));
// CHECK:  NSNumber *piFloat = ((NSNumber *(*)(Class, SEL, float))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithFloat:"), (3.14159274F));
// CHECK:  NSNumber *piDouble = ((NSNumber *(*)(Class, SEL, double))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithDouble:"), (3.1415926535000001));
// CHECK:  NSNumber *nsb = ((NSNumber *(*)(Class, SEL, BOOL))(void *)objc_msgSend)(objc_getClass("NSNumber"), sel_registerName("numberWithBool:"), (BOOL)(b));
// CHECK:  NSString *duplicateString = ((NSString *(*)(Class, SEL, const char *))(void *)objc_msgSend)(objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), (const char *)(strdup("Hello")));
