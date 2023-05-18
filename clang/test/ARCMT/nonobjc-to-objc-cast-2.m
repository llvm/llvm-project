// RUN: %clang_cc1 -arcmt-action=check -verify -triple x86_64-apple-darwin10 %s

#include "Common.h"

typedef const struct __CFString * CFStringRef;
typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);

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

@implementation NSString
-(id)string {
  if (0)
    return sref;
  else
    return strS->sref_member;
}
-(id)newString {
  return sref; // expected-error {{implicit conversion of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'id' requires a bridged cast}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
}
@end

void f(BOOL b) {
  CFStringRef cfstr;
  NSString *str = (NSString *)cfstr; // expected-error {{cast of C pointer type 'CFStringRef' (aka 'const struct __CFString *') to Objective-C pointer type 'NSString *' requires a bridged cast}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use CFBridgingRelease call to transfer ownership of a +1 'CFStringRef' (aka 'const struct __CFString *') into ARC}}
  void *vp = str;  // expected-error {{requires a bridged cast}} expected-note {{use CFBridgingRetain call}} expected-note {{use __bridge}}
}

void f2(NSString *s) {
  CFStringRef ref;
  ref = [(CFStringRef)[s string] retain]; // expected-error {{cast of Objective-C pointer type 'id' to C pointer type 'CFStringRef' (aka 'const struct __CFString *') requires a bridged cast}} \
    // expected-error {{bad receiver type 'CFStringRef' (aka 'const struct __CFString *')}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFStringRef' (aka 'const struct __CFString *')}}
}

CFStringRef f3(void) {
  return (CFStringRef)[[[NSString alloc] init] autorelease]; // expected-error {{it is not safe to cast to 'CFStringRef' the result of 'autorelease' message; a __bridge cast may result in a pointer to a destroyed object and a __bridge_retained may leak the object}} \
    // expected-note {{remove the cast and change return type of function to 'NSString *' to have the object automatically autoreleased}}
}

extern void NSLog(NSString *format, ...);

void f4(NSString *s) {
  NSLog(@"%@", (CFStringRef)s); // expected-error {{cast of Objective-C pointer type 'NSString *' to C pointer type 'CFStringRef' (aka 'const struct __CFString *') requires a bridged cast}} \
    // expected-note{{use __bridge to convert directly (no change in ownership)}} \
    // expected-note{{use CFBridgingRetain call to make an ARC object available as a +1 'CFStringRef' (aka 'const struct __CFString *')}}
}
