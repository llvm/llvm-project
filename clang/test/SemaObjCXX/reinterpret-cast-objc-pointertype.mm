// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface NSString @end

typedef const struct __CFString * CFStringRef;
const NSString* fRef;

CFStringRef func() {
  return reinterpret_cast<CFStringRef>(fRef);
}

CFStringRef fRef1;

const NSString* func1() {
  return reinterpret_cast<const NSString*>(fRef1);
}

@interface I @end
const I *fRef2;

const NSString* func2() {
  return reinterpret_cast<const NSString*>(fRef2);
}
