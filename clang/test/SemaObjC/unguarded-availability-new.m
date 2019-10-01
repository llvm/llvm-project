// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -xobjective-c++ -DMAC -triple x86_64-apple-macosx10.13 -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -Wunguarded-availability-new -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -Wno-unguarded-availability-new -DNO_WARNING -fblocks -fsyntax-only -verify %s

// unguarded-availability implies unguarded-availability-new:
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -Wunguarded-availability -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.11 -Wunguarded-availability -Wno-unguarded-availability-new -DNO_WARNING -DWARN_PREV -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -Wno-unguarded-availability -DNO_WARNING  -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -Wno-unguarded-availability -Wunguarded-availability-new -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -D TEST_FUNC_CURRENT -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.13 -D TEST_FUNC_NEXT -DNO_WARNING -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-ios11 -DNO_WARNING -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMAC -triple x86_64-apple-macosx10.12 -DWARN_CURRENT -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -DIOS -triple x86_64-apple-ios11 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DIOS -triple x86_64-apple-ios11 -D TEST_FUNC_CURRENT -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DIOS -triple x86_64-apple-ios11 -D TEST_FUNC_NEXT -DNO_WARNING -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DIOS -triple x86_64-apple-ios10.3 -DWARN_CURRENT -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -DTVOS -triple x86_64-apple-tvos11 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DTVOS -triple x86_64-apple-tvos11 -D TEST_FUNC_CURRENT -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DTVOS -triple x86_64-apple-tvos11 -D TEST_FUNC_NEXT -DNO_WARNING -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DTVOS -triple x86_64-apple-tvos10 -DWARN_CURRENT -fblocks -fsyntax-only -verify %s

// RUN: %clang_cc1 -DWATCHOS -triple i386-apple-watchos4 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DWATCHOS -triple i386-apple-watchos4 -D TEST_FUNC_CURRENT -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DWATCHOS -triple i386-apple-watchos4 -D TEST_FUNC_NEXT -DNO_WARNING -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -DWATCHOS -triple i386-apple-watchos3 -DWARN_CURRENT -fblocks -fsyntax-only -verify %s

#ifdef MAC
#define PLATFORM macos
#define NEXT 10.14

#define AVAILABLE_PREV __attribute__((availability(macos, introduced = 10.12)))
#define AVAILABLE_CURRENT __attribute__((availability(macos, introduced = 10.13)))
#define AVAILABLE_NEXT __attribute__((availability(macos, introduced = 10.14)))
#endif

#ifdef IOS
#define PLATFORM ios
#define NEXT 12

#define AVAILABLE_PREV __attribute__((availability(ios, introduced = 10)))
#define AVAILABLE_CURRENT __attribute__((availability(ios, introduced = 11)))
#define AVAILABLE_NEXT __attribute__((availability(ios, introduced = 12)))
#endif

#ifdef TVOS
#define PLATFORM tvos
#define NEXT 13

#define AVAILABLE_PREV __attribute__((availability(tvos, introduced = 10)))
#define AVAILABLE_CURRENT __attribute__((availability(tvos, introduced = 11)))
#define AVAILABLE_NEXT __attribute__((availability(tvos, introduced = 13)))
#endif

#ifdef WATCHOS
#define PLATFORM watchos
#define NEXT 5

#define AVAILABLE_PREV __attribute__((availability(watchos, introduced = 3)))
#define AVAILABLE_CURRENT __attribute__((availability(watchos, introduced = 4)))
#define AVAILABLE_NEXT __attribute__((availability(watchos, introduced = 5)))
#endif

void previouslyAvailable() AVAILABLE_PREV;
#ifdef WARN_PREV
// expected-note@-2 {{'previouslyAvailable' has been marked as being introduced}}
#endif
void currentlyAvailable() AVAILABLE_CURRENT;
#ifdef WARN_CURRENT
// expected-note@-2 {{'currentlyAvailable' has been marked as being introduced}}
#endif
void willBeAvailabile() AVAILABLE_NEXT;
#ifndef NO_WARNING
// expected-note@-2 {{'willBeAvailabile' has been marked as being introduced in}}
#endif

#ifdef TEST_FUNC_CURRENT
#define FUNC_AVAILABLE AVAILABLE_CURRENT
#endif
#ifdef TEST_FUNC_NEXT
#define FUNC_AVAILABLE AVAILABLE_NEXT
#endif
#ifndef FUNC_AVAILABLE
#define FUNC_AVAILABLE
#endif

typedef int AVAILABLE_NEXT new_int;
#ifndef NO_WARNING
// expected-note@-2 {{'new_int' has been marked as being introduced in}}
#endif
FUNC_AVAILABLE new_int x;
#ifndef NO_WARNING
#ifdef MAC
  // expected-warning@-3 {{'new_int' is only available on macOS 10.14 or newer}} expected-note@-3 {{annotate 'x' with an availability attribute to silence this warning}}
#endif
#ifdef IOS
  // expected-warning@-6 {{'new_int' is only available on iOS 12 or newer}} expected-note@-6 {{annotate 'x' with an availability attribute to silence this warning}}
#endif
#ifdef TVOS
  // expected-warning@-9 {{'new_int' is only available on tvOS 13 or newer}} expected-note@-9 {{annotate 'x' with an availability attribute to silence this warning}}
#endif
#ifdef WATCHOS
  // expected-warning@-12 {{'new_int' is only available on watchOS 5}} expected-note@-12 {{annotate 'x' with an availability attribute to silence this warning}}
#endif
#endif

void test() FUNC_AVAILABLE {
  previouslyAvailable();
#ifdef WARN_PREV
#ifdef MAC
  // expected-warning@-3 {{'previouslyAvailable' is only available on macOS 10.12 or newer}}
#endif
  // expected-note@-5 {{enclose 'previouslyAvailable' in an @available check to silence this warning}}
#endif
  currentlyAvailable();
#ifdef WARN_CURRENT
#ifdef MAC
  // expected-warning@-3 {{'currentlyAvailable' is only available on macOS 10.13 or newer}}
#endif
#ifdef IOS
  // expected-warning@-6 {{'currentlyAvailable' is only available on iOS 11 or newer}}
#endif
#ifdef TVOS
  // expected-warning@-9 {{'currentlyAvailable' is only available on tvOS 11 or newer}}
#endif
#ifdef WATCHOS
  // expected-warning@-12 {{'currentlyAvailable' is only available on watchOS 4 or newer}}
#endif
  // expected-note@-14 {{enclose 'currentlyAvailable' in an @available check to silence this warning}}
#endif
  willBeAvailabile();
#ifndef NO_WARNING
#ifdef MAC
  // expected-warning@-3 {{'willBeAvailabile' is only available on macOS 10.14 or newer}}
#endif
#ifdef IOS
  // expected-warning@-6 {{'willBeAvailabile' is only available on iOS 12 or newer}}
#endif
#ifdef TVOS
  // expected-warning@-9 {{'willBeAvailabile' is only available on tvOS 13 or newer}}
#endif
#ifdef WATCHOS
  // expected-warning@-12 {{'willBeAvailabile' is only available on watchOS 5 or newer}}
#endif
  // expected-note@-14 {{enclose 'willBeAvailabile' in an @available check to silence this warning}}
#endif
  if (@available(PLATFORM NEXT, *))
    willBeAvailabile(); // OK
}

#ifdef NO_WARNING
#ifndef WARN_PREV
// expected-no-diagnostics
#endif
#endif
