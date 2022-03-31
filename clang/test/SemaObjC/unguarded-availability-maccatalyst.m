// RUN: %clang_cc1 -triple x86_64-apple-ios14-macabi -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -xobjective-c++ -triple x86_64-apple-ios14-macabi -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -triple x86_64-apple-ios14.1-macabi -DNO_WARNING -fblocks -fsyntax-only -verify %s

#ifdef NO_WARNING
  // expected-no-diagnostics
#endif

#define AVAILABLE_PREV __attribute__((availability(macCatalyst, introduced = 13.1)))
#define AVAILABLE_CURRENT __attribute__((availability(macCatalyst, introduced = 14)))
#define AVAILABLE_NEXT __attribute__((availability(macCatalyst, introduced = 14.1)))

void previouslyAvailable(void) AVAILABLE_PREV;
void currentlyAvailable(void) AVAILABLE_CURRENT;
void willBeAvailabile(void) AVAILABLE_NEXT;
#ifndef NO_WARNING
// expected-note@-2 {{'willBeAvailabile' has been marked as being introduced in macCatalyst 14.1 here, but the deployment target is macCatalyst 14}}
#endif


typedef struct {

} Record AVAILABLE_NEXT;
#ifndef NO_WARNING
// expected-note@-2 {{'Record' has been marked as being introduced in macCatalyst 14.1 here, but the deployment target is macCatalyst 14}}
#endif

AVAILABLE_PREV
Record var;
#ifndef NO_WARNING
// expected-warning@-2 {{'Record' is only available on macCatalyst 14.1 or newer}}
// expected-note@-3 {{annotate 'var' with an availability attribute to silence this warnin}}
#endif

AVAILABLE_NEXT
Record var2;

void test(void) {
  previouslyAvailable();
  currentlyAvailable();
  willBeAvailabile();
#ifndef NO_WARNING
  // expected-warning@-2 {{'willBeAvailabile' is only available on macCatalyst 14.1 or newer}}
  // expected-note@-3 {{enclose 'willBeAvailabile' in an @available check to silence this warning}}
#endif
  if (@available(maccatalyst 14.1, *))
    willBeAvailabile(); // OK
  if (@available(ios 14.1, *))
    willBeAvailabile(); // Also OK
  if (@available(macCatalyst 14.1, *))
    willBeAvailabile(); // OK
}

void previouslyAvailableIOS(void) __attribute__((availability(ios, introduced = 10)));
void currentlyAvailableIOS(void) __attribute__((availability(ios, introduced = 14)));
void willBeAvailabileIOS(void) __attribute__((availability(ios, introduced = 14.1)));
#ifndef NO_WARNING
// expected-note@-2 {{'willBeAvailabileIOS' has been marked as being introduced in macCatalyst 14.1 here, but the deployment target is macCatalyst 14}}
#endif

void testIOSAvailabilityAlsoWorks(void) {
  previouslyAvailableIOS();
  currentlyAvailableIOS();
  willBeAvailabileIOS();
#ifndef NO_WARNING
  // expected-warning@-2 {{'willBeAvailabileIOS' is only available on macCatalyst 14.1 or newer}}
  // expected-note@-3 {{enclose 'willBeAvailabileIOS' in an @available check to silence this warning}}
#endif
  if (@available(macCatalyst 14.1, *))
    willBeAvailabileIOS(); // OK
  if (@available(ios 14.1, *))
    willBeAvailabile(); // Also OK
}

typedef struct {

} Record2 __attribute__((availability(ios, introduced = 14.1)));
#ifndef NO_WARNING
// expected-note@-2 {{'Record2' has been marked as being introduced in macCatalyst 14.1 here, but the deployment target is macCatalyst 14}}
#endif

__attribute__((availability(ios, introduced = 10)))
Record2 var11;
#ifndef NO_WARNING
// expected-warning@-2 {{'Record2' is only available on macCatalyst 14.1 or newer}}
// expected-note@-3 {{annotate 'var11' with an availability attribute to silence this warnin}}
#endif

__attribute__((availability(ios, introduced = 14.1)))
Record2 var12;
