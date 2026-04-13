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
    // expected-warning@-1{{Call argument is unretained and unsafe}}
  consumeNSString(NSApplicationDidBecomeActiveNotification);
  consumeNSString(LocalGlobalNSString);
    // expected-warning@-1{{Call argument is unretained and unsafe}}
}
