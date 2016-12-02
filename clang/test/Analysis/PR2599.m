// RUN: %clang_cc1 -triple %itanium_abi_triple -analyze -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -analyzer-store=region -fobjc-gc -verify %s

typedef const void * CFTypeRef;
typedef const struct __CFString * CFStringRef;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFDictionary * CFDictionaryRef;
CFTypeRef CFMakeCollectable(CFTypeRef cf) ;
extern CFStringRef CFStringCreateWithFormat(CFAllocatorRef alloc, CFDictionaryRef formatOptions, CFStringRef format, ...);
typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol
NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
@end
enum { NSASCIIStringEncoding = 1,     NSNEXTSTEPStringEncoding = 2,     NSJapaneseEUCStringEncoding = 3,     NSUTF8StringEncoding = 4,     NSISOLatin1StringEncoding = 5,     NSSymbolStringEncoding = 6,     NSNonLossyASCIIStringEncoding = 7,     NSShiftJISStringEncoding = 8,     NSISOLatin2StringEncoding = 9,     NSUnicodeStringEncoding = 10,     NSWindowsCP1251StringEncoding = 11,     NSWindowsCP1252StringEncoding = 12,     NSWindowsCP1253StringEncoding = 13,     NSWindowsCP1254StringEncoding = 14,     NSWindowsCP1250StringEncoding = 15,     NSISO2022JPStringEncoding = 21,     NSMacOSRomanStringEncoding = 30,      NSUTF16StringEncoding = NSUnicodeStringEncoding,       NSUTF16BigEndianStringEncoding = 0x90000100,     NSUTF16LittleEndianStringEncoding = 0x94000100,      NSUTF32StringEncoding = 0x8c000100,     NSUTF32BigEndianStringEncoding = 0x98000100,     NSUTF32LittleEndianStringEncoding = 0x9c000100  };
typedef NSUInteger NSStringEncoding;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
- (id)initWithBytesNoCopy:(void *)bytes length:(NSUInteger)len encoding:(NSStringEncoding)encoding freeWhenDone:(BOOL)freeBuffer;
@end
@interface NSAutoreleasePool : NSObject {}
- (void)drain;
@end
extern NSString * const NSXMLParserErrorDomain ;

// The actual test case.  UTIL_AUTORELEASE_CF_AS_ID is a macro that doesn't
// actually do what it was intended to.

#define NSSTRINGWRAPPER(bytes,len) \
  [[[NSString alloc] initWithBytesNoCopy: (void*)(bytes) length: (len) encoding: NSUTF8StringEncoding freeWhenDone: (BOOL)0] autorelease]

#define UTIL_AUTORELEASE_CF_AS_ID(cf) ( (((void*)0) == (cf)) ? ((void*)0) : [(id) CFMakeCollectable( (CFTypeRef) cf) autorelease] )

#define UTIL_AUTORELEASE_CF_AS_ID_WITHOUT_TEST(cf) ( [(id) CFMakeCollectable( (CFTypeRef) cf) autorelease] )

static char *lorem = "fooBarBaz";

void NSLog(NSString *, ...);

int main (int argc, const char * argv[]) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  NSString *tmp1 = NSSTRINGWRAPPER(lorem, 6); // no-warning
  NSString *tmp2 = UTIL_AUTORELEASE_CF_AS_ID( CFStringCreateWithFormat(((void*)0), ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "lorem: %@" "")), tmp1) );  // expected-warning 2 {{leak}}
  NSString *tmp3 = UTIL_AUTORELEASE_CF_AS_ID_WITHOUT_TEST( CFStringCreateWithFormat(((void*)0), ((void*)0), ((CFStringRef) __builtin___CFStringMakeConstantString ("" "lorem: %@" "")), tmp1) );
  NSLog(@"tmp2: %@ tmp3: %@", tmp2, tmp3);
  [pool drain];
  return 0;
}
