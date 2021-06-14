// RUN: %clang_cc1 -triple x86_64-apple-darwin -Wformat-nonliteral -fsyntax-only -fblocks -verify -Wno-objc-root-class %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not including Foundation.h directly makes this test case both svelt and
// portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

#include <stdarg.h>

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef long NSInteger;
@class NSString, Protocol;
extern void NSLog(NSString *format, ...);
extern void NSLogv(NSString *format, va_list args);
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
typedef float CGFloat;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+(instancetype)stringWithFormat:(NSString *)fmt, ...
    __attribute__((format(__NSString__, 1, 2)));
@end
@interface NSSimpleCString : NSString {} @end
@interface NSConstantString : NSSimpleCString @end
extern void *_NSConstantStringClassReference;

@interface NSAttributedString : NSObject
+(instancetype)stringWithFormat:(NSAttributedString *)fmt, ...
    __attribute__((format(__NSString__, 1, 2)));
@end

typedef const struct __CFString * CFStringRef;
extern void CFStringCreateWithFormat(CFStringRef format, ...) __attribute__((format(CFString, 1, 2)));
#define CFSTR(cStr)  ((CFStringRef) __builtin___CFStringMakeConstantString ("" cStr ""))

// This function is used instead of the builtin if -fno-constant-cfstrings.
// The definition on Mac OS X is NOT annotated with format_arg as of 10.8,
// but clang will implicitly add the attribute if it's not written.
extern CFStringRef __CFStringMakeConstantString(const char *);

int printf(const char * restrict, ...) ;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

void check_nslog(unsigned k) {
  NSLog(@"%d%%", k); // no-warning
  NSLog(@"%s%lb%d", "unix", 10, 20); // expected-warning {{invalid conversion specifier 'b'}} expected-warning {{data argument not used by format string}}
}

// Check type validation
extern void NSLog2(int format, ...) __attribute__((format(__NSString__, 1, 2))); // expected-error {{format argument not an NSString}}
extern void CFStringCreateWithFormat2(int *format, ...) __attribute__((format(CFString, 1, 2))); // expected-error {{format argument not a CFString}}

// <rdar://problem/7068334> - Catch use of long long with int arguments.
void rdar_7068334() {
  long long test = 500;  
  printf("%i ",test); // expected-warning{{format specifies type 'int' but the argument has type 'long long'}}
  NSLog(@"%i ",test); // expected-warning{{format specifies type 'int' but the argument has type 'long long'}}
  CFStringCreateWithFormat(CFSTR("%i"),test); // expected-warning{{format specifies type 'int' but the argument has type 'long long'}}
}

// <rdar://problem/7697748>
void rdar_7697748() {
  NSLog(@"%@!"); // expected-warning{{more '%' conversions than data arguments}}
}

@protocol Foo;

void test_p_conversion_with_objc_pointer(id x, id<Foo> y) {
  printf("%p", x); // no-warning
  printf("%p", y); // no-warning
}

// <rdar://problem/10696348>, PR 10274 - CFString and NSString formats are ignored
extern void MyNSLog(NSString *format, ...) __attribute__((format(__NSString__, 1, 2)));
extern void MyCFStringCreateWithFormat(CFStringRef format, ...) __attribute__((format(__CFString__, 1, 2)));

void check_mylog() {
  MyNSLog(@"%@"); // expected-warning {{more '%' conversions than data arguments}}
  MyCFStringCreateWithFormat(CFSTR("%@")); // expected-warning {{more '%' conversions than data arguments}}
}

// PR 10275 - format function attribute isn't checked in Objective-C methods
@interface Foo
+ (id)fooWithFormat:(NSString *)fmt, ... __attribute__((format(__NSString__, 1, 2)));
+ (id)fooWithCStringFormat:(const char *)format, ... __attribute__((format(__printf__, 1, 2)));
@end

void check_method() {
  [Foo fooWithFormat:@"%@"]; // expected-warning {{more '%' conversions than data arguments}}
  [Foo fooWithCStringFormat:"%@"]; // expected-warning {{invalid conversion specifier '@'}}
}

// Warn about using BOOL with %@
void rdar10743758(id x) {
  NSLog(@"%@ %@", x, (BOOL) 1); // expected-warning {{format specifies type 'id' but the argument has type 'BOOL' (aka 'signed char')}}
}

NSString *test_literal_propagation(void) {
  const char * const s1 = "constant string %s"; // expected-note {{format string is defined here}}
  printf(s1); // expected-warning {{more '%' conversions than data arguments}}
  const char * const s5 = "constant string %s"; // expected-note {{format string is defined here}}
  const char * const s2 = s5;
  printf(s2); // expected-warning {{more '%' conversions than data arguments}}

  const char * const s3 = (const char *)0;
  printf(s3); // no-warning (NULL is a valid format string)

  NSString * const ns1 = @"constant string %s"; // expected-note {{format string is defined here}}
  NSLog(ns1); // expected-warning {{more '%' conversions than data arguments}}
  NSString * const ns5 = @"constant string %s"; // expected-note {{format string is defined here}}
  NSString * const ns2 = ns5;
  NSLog(ns2); // expected-warning {{more '%' conversions than data arguments}}
  NSString * ns3 = ns1;
  NSLog(ns3); // expected-warning {{format string is not a string literal}}}
  // expected-note@-1{{treat the string as an argument to avoid this}}

  NSString * const ns6 = @"split" " string " @"%s"; // expected-note {{format string is defined here}}
  NSLog(ns6); // expected-warning {{more '%' conversions than data arguments}}
}

// Do not emit warnings when using NSLocalizedString
#include "format-strings-system.h"

// Test it inhibits diag only for macros in system headers
#define MyNSLocalizedString(key) GetLocalizedString(key)
#define MyNSAssert(fmt, arg) NSLog(fmt, arg, 0, 0)

void check_NSLocalizedString() {
  [Foo fooWithFormat:NSLocalizedString(@"format"), @"arg"]; // no-warning
  [Foo fooWithFormat:MyNSLocalizedString(@"format"), @"arg"]; // expected-warning {{format string is not a string literal}}}
}

void check_NSAssert() {
  NSAssert(@"Hello %@", @"World"); // no-warning
  MyNSAssert(@"Hello %@", @"World"); // expected-warning  {{data argument not used by format string}}
}

typedef __WCHAR_TYPE__ wchar_t;

// Test that %S, %C, %ls check for 16 bit types in ObjC strings, as described at
// http://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/Strings/Articles/formatSpecifiers.html#//apple_ref/doc/uid/TP40004265

void test_percent_S() {
  const unsigned short data[] = { 'a', 'b', 0 };
  const unsigned short* ptr = data;
  NSLog(@"%S", ptr);  // no-warning

  const wchar_t* wchar_ptr = L"ab";
  NSLog(@"%S", wchar_ptr);  // expected-warning{{format specifies type 'const unichar *' (aka 'const unsigned short *') but the argument has type 'const wchar_t *'}}
}

void test_percent_ls() {
  const unsigned short data[] = { 'a', 'b', 0 };
  const unsigned short* ptr = data;
  NSLog(@"%ls", ptr);  // no-warning

  const wchar_t* wchar_ptr = L"ab";
  NSLog(@"%ls", wchar_ptr);  // expected-warning{{format specifies type 'const unichar *' (aka 'const unsigned short *') but the argument has type 'const wchar_t *'}}
}

void test_percent_C() {
  const unsigned short data = 'a';
  NSLog(@"%C", data);  // no-warning

  const wchar_t wchar_data = L'a';
  NSLog(@"%C", wchar_data);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'wchar_t'}}
}

// Test that %@ works with toll-free bridging (<rdar://problem/10814120>).
void test_toll_free_bridging(CFStringRef x, id y) {
  NSLog(@"%@", x); // no-warning
  CFStringCreateWithFormat(CFSTR("%@"), x); // no-warning

  NSLog(@"%@", y); // no-warning
  CFStringCreateWithFormat(CFSTR("%@"), y); // no-warning
}

@interface Bar
+ (void)log:(NSString *)fmt, ...;
+ (void)log2:(NSString *)fmt, ... __attribute__((format(NSString, 1, 2)));
@end

@implementation Bar

+ (void)log:(NSString *)fmt, ... {
  va_list ap;
  va_start(ap,fmt);
  NSLogv(fmt, ap); // expected-warning{{format string is not a string literal}}
  va_end(ap);
}

+ (void)log2:(NSString *)fmt, ... {
  va_list ap;
  va_start(ap,fmt);
  NSLogv(fmt, ap); // no-warning
  va_end(ap);
}

@end


// Test that it is okay to use %p with the address of a block.
void rdar11049844_aux();
int rdar11049844() {
  typedef void (^MyBlock)(void);
  MyBlock x = ^void() { rdar11049844_aux(); };
  printf("%p", x);  // no-warning
}

void test_nonBuiltinCFStrings() {
  CFStringCreateWithFormat(__CFStringMakeConstantString("%@"), 1); // expected-warning{{format specifies type 'id' but the argument has type 'int'}}
}


// Don't crash on an invalid argument expression.
// <rdar://problem/11890818>
@interface NSDictionary : NSObject
- (id)objectForKeyedSubscript:(id)key;
@end

void testInvalidFormatArgument(NSDictionary *dict) {
  NSLog(@"no specifiers", dict[CFSTR("abc")]); // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}
  NSLog(@"%@", dict[CFSTR("abc")]); // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}
  NSLog(@"%@ %@", dict[CFSTR("abc")]); // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}

  [Foo fooWithFormat:@"no specifiers", dict[CFSTR("abc")]]; // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}
  [Foo fooWithFormat:@"%@", dict[CFSTR("abc")]]; // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}}
  [Foo fooWithFormat:@"%@ %@", dict[CFSTR("abc")]]; // expected-error{{indexing expression is invalid because subscript type 'CFStringRef' (aka 'const struct __CFString *') is not an integral or Objective-C pointer type}} expected-warning{{more '%' conversions than data arguments}}
}


// <rdar://problem/11825593>
void testByValueObjectInFormat(Foo *obj) {
  printf("%d %d %d", 1L, *obj, 1L); // expected-error {{cannot pass object with interface type 'Foo' by value to variadic function; expected type from format string was 'int'}} expected-warning 2 {{format specifies type 'int' but the argument has type 'long'}}
  printf("%!", *obj); // expected-error {{cannot pass object with interface type 'Foo' by value through variadic function}} expected-warning {{invalid conversion specifier}}
  printf(0, *obj); // expected-error {{cannot pass object with interface type 'Foo' by value through variadic function}}

  [Bar log2:@"%d", *obj]; // expected-error {{cannot pass object with interface type 'Foo' by value to variadic method; expected type from format string was 'int'}}
}

// <rdar://problem/13557053>
void testTypeOf(NSInteger dW, NSInteger dH) {
  NSLog(@"dW %d  dH %d",({ __typeof__(dW) __a = (dW); __a < 0 ? -__a : __a; }),({ __typeof__(dH) __a = (dH); __a < 0 ? -__a : __a; })); // expected-warning 2 {{format specifies type 'int' but the argument has type 'long'}}
}

void testUnicode() {
  NSLog(@"%C", 0x2022); // no-warning
  NSLog(@"%C", 0x202200); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
}

// Test Objective-C modifier flags.
void testObjCModifierFlags() {
  NSLog(@"%[]@", @"Foo"); // expected-warning {{missing object format flag}}
  NSLog(@"%[", @"Foo"); // expected-warning {{incomplete format specifier}}
  NSLog(@"%[tt", @"Foo");  // expected-warning {{incomplete format specifier}}
  NSLog(@"%[tt]@", @"Foo"); // no-warning
  NSLog(@"%[tt]@ %s", @"Foo", "hello"); // no-warning
  NSLog(@"%s %[tt]@", "hello", @"Foo"); // no-warning
  NSLog(@"%[blark]@", @"Foo"); // expected-warning {{'blark' is not a valid object format flag}}
  NSLog(@"%2$[tt]@ %1$[tt]@", @"Foo", @"Bar"); // no-warning
  NSLog(@"%2$[tt]@ %1$[tt]s", @"Foo", @"Bar"); // expected-warning {{object format flags cannot be used with 's' conversion specifier}}
}

// rdar://23622446
@interface RD23622446_Tester: NSObject

+ (void)stringWithFormat:(const char *)format, ... __attribute__((format(__printf__, 1, 2)));

@end

@implementation RD23622446_Tester

__attribute__ ((format_arg(1)))
const char *rd23622446(const char *format) {
  return format;
}

+ (void)stringWithFormat:(const char *)format, ... {
  return;
}

- (const char *)test:(const char *)format __attribute__ ((format_arg(1))) {
  return format;
}

- (NSString *)str:(NSString *)format __attribute__ ((format_arg(1))) {
  return format;
}

- (void)foo {
  [RD23622446_Tester stringWithFormat:rd23622446("%u"), 1, 2]; // expected-warning {{data argument not used by format string}}
  [RD23622446_Tester stringWithFormat:[self test: "%u"], 1, 2]; // expected-warning {{data argument not used by format string}}
  [RD23622446_Tester stringWithFormat:[self test: "%s %s"], "name"]; // expected-warning {{more '%' conversions than data arguments}}
  NSLog([self str: @"%@ %@"], @"name"); // expected-warning {{more '%' conversions than data arguments}}
  [RD23622446_Tester stringWithFormat:rd23622446("%d"), 1]; // ok
  [RD23622446_Tester stringWithFormat:[self test: "%d %d"], 1, 2]; // ok
  NSLog([self str: @"%@"], @"string"); // ok
}

@end

@interface NSBundle : NSObject
- (NSString *)localizedStringForKey:(NSString *)key
                              value:(nullable NSString *)value
                              table:(nullable NSString *)tableName
     __attribute__((format_arg(1)));

- (NSString *)someRandomMethod:(NSString *)key
                         value:(nullable NSString *)value
                         table:(nullable NSString *)tableName
    __attribute__((format_arg(1)));

- (NSAttributedString *)someMethod2:(NSString *)key
    __attribute__((format_arg(1)));
@end

void useLocalizedStringForKey(NSBundle *bndl) {
  [NSString stringWithFormat:
              [bndl localizedStringForKey:@"%d" // expected-warning{{more '%' conversions than data arguments}}
                                      value:0
                                      table:0]];
  // No warning, @"flerp" doesn't have a format specifier.
  [NSString stringWithFormat: [bndl localizedStringForKey:@"flerp" value:0 table:0], 43, @"flarp"];

  [NSString stringWithFormat:
              [bndl localizedStringForKey:@"%f"
                                    value:0
                                    table:0], 42]; // expected-warning{{format specifies type 'double' but the argument has type 'int'}}

  [NSString stringWithFormat:
              [bndl someRandomMethod:@"%f"
                               value:0
                               table:0], 42]; // expected-warning{{format specifies type 'double' but the argument has type 'int'}}

  [NSString stringWithFormat:
              [bndl someRandomMethod:@"flerp"
                               value:0
                               table:0], 42]; // expected-warning{{data argument not used by format string}}

  [NSAttributedString stringWithFormat:
              [bndl someMethod2: @"test"], 5]; // expected-warning{{data argument not used by format string}}
  [NSAttributedString stringWithFormat:
              [bndl someMethod2: @"%f"], 42]; // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
}
