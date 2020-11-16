// RUN: %clang_cc1 -arcmt-action=check -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c -verify %s
// rdar://10387088
typedef const void * CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);

extern 
CFTypeRef CFRetain(CFTypeRef cf);

@interface INTF
{
  void *cf_format;
  id objc_format;
}
@end

@interface NSString
+ (id)stringWithFormat:(NSString *)format;
@end

@implementation INTF
- (void) Meth {
  NSString *result;

  result = (id) CFRetain([NSString stringWithFormat:@"PBXLoopMode"]); // expected-error {{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
								      // expected-note {{use __bridge to convert directly (no change in ownership)}} \
								      // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}

  result = (id) CFRetain((id)((objc_format))); // expected-error {{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
					       // expected-note {{use __bridge to convert directly (no change in ownership)}} \
					       // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}

  result = (id) CFRetain((id)((cf_format))); // expected-error {{cast of C pointer type 'CFTypeRef' (aka 'const void *') to Objective-C pointer type 'id' requires a bridged cast}} \
					     // expected-note {{use __bridge to convert directly (no change in ownership)}} \
                                             // expected-note {{use CFBridgingRelease call to transfer ownership of a +1 'CFTypeRef' (aka 'const void *') into ARC}}

  result = (id) CFRetain((CFTypeRef)((objc_format)));

  result = (id) CFRetain(cf_format); // OK
}
@end

