// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result
// DISABLE: mingw32

#include "Common.h"

typedef const struct __CFString * CFStringRef;
extern const CFStringRef kUTTypePlainText;
extern const CFStringRef kUTTypeRTF;

typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFUUID * CFUUIDRef;

extern const CFAllocatorRef kCFAllocatorDefault;

extern CFStringRef CFUUIDCreateString(CFAllocatorRef alloc, CFUUIDRef uuid);

struct StrS {
  CFStringRef sref_member;
};

@interface NSString : NSObject {
  CFStringRef sref;
  struct StrS *strS;
}
-(id)string;
-(id)newString;
@end

void f(BOOL b, id p) {
  NSString *str = (NSString *)kUTTypePlainText;
  str = b ? kUTTypeRTF : kUTTypePlainText;
  str = (NSString *)(b ? kUTTypeRTF : kUTTypePlainText);
  str = (NSString *)p; // no change.

  CFUUIDRef   _uuid;
  NSString *_uuidString = (NSString *)CFUUIDCreateString(kCFAllocatorDefault, _uuid);
  _uuidString = [(NSString *)CFUUIDCreateString(kCFAllocatorDefault, _uuid) autorelease];
  _uuidString = CFRetain(_uuid);
}

@implementation NSString (StrExt)
- (NSString *)stringEscapedAsURI {
  CFStringRef str = (CFStringRef)self;
  CFStringRef str2 = self;
  return self;
}
@end

@implementation NSString
-(id)string {
  if (0)
    return sref;
  else
    return strS->sref_member;
}
-(id)newString { return 0; }
@end

extern void consumeParam(CFStringRef CF_CONSUMED p);

void f2(NSString *s) {
  CFStringRef ref = [s string];
  ref = (CFStringRef)[s string];
  ref = s.string;
  ref = [NSString new];
  ref = [s newString];
  ref = (CFStringRef)[NSString new];
  ref = [[NSString alloc] init];
  ref = [[s string] retain];
  ref = CFRetain((CFStringRef)[s string]);
  ref = CFRetain([s string]);
  ref = CFRetain(s);
  ref = [s retain];

  consumeParam((CFStringRef)s);
  consumeParam(s);
}
