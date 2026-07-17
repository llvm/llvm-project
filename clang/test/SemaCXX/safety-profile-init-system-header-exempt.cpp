// Temporary system-header exemption for profile enforcement (stopgap for the
// not-yet-implemented [[profiles::exempt]], P3589R2 s1.1.6). By default a
// violation originating in a system header is exempt; a violation in a normal
// (-I) header is still enforced. -fno-profiles-exempt-system-headers enforces
// profiles in system-header code too.

// RUN: rm -rf %t
// RUN: split-file %s %t
//
// Default: exemption on -- only the -I header's violation fires.
// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 -I %t %t/main.cpp
//
// Exemption off: both the system-header and the -I header violations fire.
// RUN: %clang_cc1 -fsyntax-only -verify=expected,strict -fno-profiles-exempt-system-headers -fprofiles -std=c++23 -I %t %t/main.cpp

//--- user.h
// A normal header reached via -I is not a system header, so it is enforced in
// both runs.
inline void user_fn() {
  int y; // expected-error {{variable 'y' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)y;
}

//--- sys.h
#pragma clang system_header
// This is a system header: exempt by default, enforced only with
// -fno-profiles-exempt-system-headers.
inline void sys_fn() {
  int x; // strict-error {{variable 'x' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)x;
}

//--- main.cpp
[[profiles::enforce(std::init)]];
#include "user.h"
#include "sys.h"
