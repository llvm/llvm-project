// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedCallArgsChecker -verify %s

#import "mock-types.h"
#import "mock-system-header.h"

void consumeCFString(CFStringRef);
extern "C" CFStringRef LocalGlobalCFString;
void consumeNSString(NSString *);
extern "C" NSString *LocalGlobalNSString;

void foo() {
  consumeCFString(kCFURLTagNamesKey);
  consumeCFString(LocalGlobalCFString);
    // expected-warning@-1{{Function argument 'LocalGlobalCFString' (to 'consumeCFString') is a RetainPtr-capable type 'CFStringRef'}}
  consumeNSString(NSApplicationDidBecomeActiveNotification);
  consumeNSString(LocalGlobalNSString);
    // expected-warning@-1{{Function argument 'LocalGlobalNSString' (to 'consumeNSString') is a raw pointer to RetainPtr-capable type 'NSString'}}
}
