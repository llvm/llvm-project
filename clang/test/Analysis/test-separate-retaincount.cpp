// RUN: %clang_analyze_cc1 -std=c++14 -verify=no-retain-count %s \
// RUN:   -analyzer-checker=core,osx \
// RUN:   -analyzer-disable-checker osx.cocoa.RetainCount
//
// RUN: %clang_analyze_cc1 -std=c++14 -verify=no-os-object %s \
// RUN:   -analyzer-checker=core,osx \
// RUN:   -analyzer-disable-checker osx.OSObjectRetainCount

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
  CFRelease(cf); // no-os-object-warning{{Reference-counted object is used after it is released}}
}

void osobject_overrelease() {
  OSObject *o = new OSObject;
  o->release();
  o->release(); // no-retain-count-warning{{Reference-counted object is used after it is released}}
}
