// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify %s

#if __has_feature(arc_cf_code_audited)
char _global[-1]; // expected-error {{declared as an array with a negative size}}
#endif

typedef const void *CFTypeRef;
CFTypeRef CFBridgingRetain(id X);
id CFBridgingRelease(CFTypeRef);
typedef const struct __CFString *CFStringRef;

extern CFStringRef CFMakeString0(void);
#pragma clang arc_cf_code_audited begin
extern CFStringRef CFCreateString0(void);
#pragma clang arc_cf_code_audited end
void test0() {
  id x;
  x = (id) CFMakeString0(); // expected-error {{requires a bridged cast}} expected-note {{__bridge to convert directly}} expected-note {{CFBridgingRelease call to transfer}}
  x = (id) CFCreateString0(); // expected-error {{requires a bridged cast}} expected-note {{CFBridgingRelease call to transfer}}
}

extern CFStringRef CFMakeString1(void) __attribute__((cf_returns_not_retained));
extern CFStringRef CFCreateString1(void) __attribute__((cf_returns_retained));
void test1() {
  id x;
  x = (id) CFMakeString1();
  x = (id) CFCreateString1(); // expected-error {{requires a bridged cast}} expected-note {{CFBridgingRelease call to transfer}}
}

#define CF_AUDIT_BEGIN _Pragma("clang arc_cf_code_audited begin")
#define CF_AUDIT_END _Pragma("clang arc_cf_code_audited end")
#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))

CF_AUDIT_BEGIN
extern CFStringRef CFMakeString2(void);
extern CFStringRef CFCreateString2(void) CF_RETURNS_NOT_RETAINED;
extern CFStringRef CFMakeString3(void) CF_RETURNS_RETAINED;
extern CFStringRef CFCreateString3(void);
CF_AUDIT_END
void test2() {
  id x;
  x = (id) CFMakeString2();
  x = (id) CFCreateString2();
  x = (id) CFMakeString3(); // expected-error {{requires a bridged cast}} expected-note {{CFBridgingRelease call to transfer}}
  x = (id) CFCreateString3(); // expected-error {{requires a bridged cast}} expected-note {{CFBridgingRelease call to transfer}}
}
