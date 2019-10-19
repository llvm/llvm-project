// RUN: %clang_analyze_cc1 -std=c++14 -DNO_CF_OBJECT -verify %s \
// RUN:   -analyzer-checker=core,osx \
// RUN:   -analyzer-disable-checker osx.cocoa.RetainCount
//
// RUN: %clang_analyze_cc1 -std=c++14 -DNO_OS_OBJECT -verify %s \
// RUN:   -analyzer-checker=core,osx \
// RUN:   -analyzer-disable-checker osx.OSObjectRetainCount
//
// RUN: %clang_analyze_cc1 -std=c++14 -DNO_OS_OBJECT -verify %s \
// RUN:   -analyzer-checker=core,osx \
// RUN:   -analyzer-config "osx.cocoa.RetainCount:CheckOSObject=false"

#include "os_object_base.h"

typedef const void * CFTypeRef;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);

#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
extern CFTypeRef CFCreate() CF_RETURNS_RETAINED;

using size_t = decltype(sizeof(int));

void cf_overrelease() {
  CFTypeRef cf = CFCreate();
  CFRelease(cf);
  CFRelease(cf);
#ifndef NO_CF_OBJECT
  // expected-warning@-2{{Reference-counted object is used after it is released}}
#endif
}

void osobject_overrelease() {
  OSObject *o = new OSObject;
  o->release();
  o->release();
#ifndef NO_OS_OBJECT
  // expected-warning@-2{{Reference-counted object is used after it is released}}
#endif
}
