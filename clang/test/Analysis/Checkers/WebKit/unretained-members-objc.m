// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker -verify %s

typedef const struct __attribute__((objc_bridge(NSString))) __CFString *CFStringRef;

__attribute__((objc_root_class))
@interface NSObject
@end

@interface PlainObjCMembers : NSObject {
  CFStringRef _cf;
  // expected-warning@-1{{Instance variable '_cf' (of 'PlainObjCMembers') is a RetainPtr-capable type 'CFStringRef'}}
  NSObject *_ns;
  // expected-warning@-1{{Instance variable '_ns' (of 'PlainObjCMembers') is a raw pointer to RetainPtr-capable type 'NSObject'}}
  id _id;
  // expected-warning@-1{{Instance variable '_id' (of 'PlainObjCMembers') is a RetainPtr-capable type 'id'}}
}
@end

@implementation PlainObjCMembers
@end
