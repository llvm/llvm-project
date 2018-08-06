// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.RetainCount -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.coreFoundation.CFRetainRelease,osx.cocoa.RetainCount -analyzer-inline-max-stack-depth=0 -verify %s

#pragma clang arc_cf_code_audited begin
typedef const void * CFTypeRef;
extern CFTypeRef CFRetain(CFTypeRef cf);
extern void CFRelease(CFTypeRef cf);
#pragma clang arc_cf_code_audited end

#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#define CF_CONSUMED __attribute__((cf_consumed))

extern CFTypeRef CFCreate() CF_RETURNS_RETAINED;

// A "safe" variant of CFRetain that doesn't crash when a null pointer is
// retained. This is often defined by users in a similar manner. The
// CF_RETURNS_RETAINED annotation is misleading here, because the function
// is not supposed to return an object with a +1 retain count. Instead, it
// is supposed to return an object with +(N+1) retain count, where N is
// the original retain count of 'cf'. However, there is no good annotation
// to use in this case, and it is pointless to provide such annotation
// because the only use cases would be CFRetain and SafeCFRetain.
// So instead we teach the analyzer to be able to accept such code
// and ignore the misplaced annotation.
CFTypeRef SafeCFRetain(CFTypeRef cf) CF_RETURNS_RETAINED {
  if (cf) {
    return CFRetain(cf);
  }
  return cf;
}

// A "safe" variant of CFRelease that doesn't crash when a null pointer is
// released. The CF_CONSUMED annotation seems reasonable here.
void SafeCFRelease(CFTypeRef CF_CONSUMED cf) {
  if (cf)
    CFRelease(cf); // no-warning (when inlined)
}

// The same thing, just with a different naming style.
CFTypeRef retainCFType(CFTypeRef cf) CF_RETURNS_RETAINED {
  if (cf) {
    return CFRetain(cf);
  }
  return cf;
}

void releaseCFType(CFTypeRef CF_CONSUMED cf) {
  if (cf)
    CFRelease(cf); // no-warning (when inlined)
}

void escape(CFTypeRef cf);

void makeSureTestsWork() {
  CFTypeRef cf = CFCreate();
  CFRelease(cf);
  CFRelease(cf); // expected-warning{{Reference-counted object is used after it is released}}
}

// Make sure we understand that the second SafeCFRetain doesn't return an
// object with +1 retain count, which we won't be able to release twice.
void falseOverrelease(CFTypeRef cf) {
  SafeCFRetain(cf);
  SafeCFRetain(cf);
  SafeCFRelease(cf);
  SafeCFRelease(cf); // no-warning after inlining this.
}

// Regular CFRelease() should behave similarly.
void sameWithNormalRelease(CFTypeRef cf) {
  SafeCFRetain(cf);
  SafeCFRetain(cf);
  CFRelease(cf);
  CFRelease(cf); // no-warning
}

// Make sure we understand that the second SafeCFRetain doesn't return an
// object with +1 retain count, which would no longer be owned by us after
// it escapes to escape() and released once.
void falseReleaseNotOwned(CFTypeRef cf) {
  SafeCFRetain(cf);
  SafeCFRetain(cf);
  escape(cf);
  SafeCFRelease(cf);
  SafeCFRelease(cf); // no-warning after inlining this.
}

void testTheOtherNamingConvention(CFTypeRef cf) {
  retainCFType(cf);
  retainCFType(cf);
  releaseCFType(cf);
  releaseCFType(cf); // no-warning
}
