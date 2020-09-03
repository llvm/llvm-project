// RUN: %clang_analyze_cc1 -verify -Wno-objc-root-class %s \
// RUN:   -analyzer-checker=core,osx.cocoa.RetainCount

#define NULL 0
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#define CF_CONSUMED __attribute__((cf_consumed))

void clang_analyzer_eval(int);

typedef const void *CFTypeRef;

extern CFTypeRef CFCreate() CF_RETURNS_RETAINED;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);

void bar(CFTypeRef *v) {}

void test1() {
  CFTypeRef *values = (CFTypeRef[]){
      CFCreate(),  // no-warning
      CFCreate(),  // expected-warning{{leak}}
      CFCreate()}; // no-warning
  CFRelease(values[0]);
  CFRelease(values[2]);
}
