// RUN: %clang_analyze_cc1 -Wno-objc-literal-conversion -analyzer-checker=core,unix.Malloc,osx.cocoa.NonNilReturnValue,debug.ExprInspection -analyzer-store=region -verify %s

void clang_analyzer_eval(int);

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
    return @(strdup("boxed dynamic string")); // expected-warning{{Potential memory leak}}
}

id const_char_pointer(int *x) {
  if (x)
    return @(3);
  return @(*x); // expected-warning {{Dereference of null pointer (loaded from variable 'x')}}
}

void checkNonNil() {
  clang_analyzer_eval(!!@3); // expected-warning{{TRUE}}
  clang_analyzer_eval(!!@(3+4)); // expected-warning{{TRUE}}
  clang_analyzer_eval(!!@(57.0)); // expected-warning{{TRUE}}

  const char *str = "abc";
  clang_analyzer_eval(!!@(str)); // expected-warning{{TRUE}}
  clang_analyzer_eval(!!@__objc_yes); // expected-warning{{TRUE}}
}

