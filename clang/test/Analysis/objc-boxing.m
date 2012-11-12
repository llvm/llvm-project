// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -analyzer-store=region -verify %s

typedef signed char BOOL;
typedef long NSInteger;
typedef unsigned long NSUInteger;
@interface NSString @end
@interface NSString (NSStringExtensionMethods)
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end

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


extern char *strdup(const char *str);

id constant_string() {
    return @("boxed constant string.");
}

id dynamic_string() {
    return @(strdup("boxed dynamic string")); // expected-warning{{Memory is never released; potential leak}}
}

id const_char_pointer(int *x) {
  if (x)
    return @(3);
  return @(*x); // expected-warning {{Dereference of null pointer (loaded from variable 'x')}}
}