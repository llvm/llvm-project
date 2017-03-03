// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.RetainCount -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,osx.cocoa.RetainCount -verify %s -x objective-c++

// The special thing about this file is that CFRetain and CFRelease are marked
// as cf_audited_transfer.

#pragma clang arc_cf_code_audited begin
typedef const void * CFTypeRef;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);

extern CFTypeRef CFCreateSomethingAudited();
#pragma clang arc_cf_code_audited end

extern CFTypeRef CFCreateSomethingUnaudited();

void testAudited() {
  CFTypeRef obj = CFCreateSomethingAudited(); // no-warning
  CFRelease(obj); // no-warning

  CFTypeRef obj2 = CFCreateSomethingAudited(); // expected-warning{{leak}}
  CFRetain(obj2); // no-warning
  CFRelease(obj2); // no-warning
}

void testUnaudited() {
  CFTypeRef obj = CFCreateSomethingUnaudited(); // no-warning
  CFRelease(obj); // no-warning

  CFTypeRef obj2 = CFCreateSomethingUnaudited(); // expected-warning{{leak}}
  CFRetain(obj2); // no-warning
  CFRelease(obj2); // no-warning
}
