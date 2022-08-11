// RUN: %clang_cc1 -std=c++1z %s -verify
// RUN: %clang_cc1 -std=c++1z %s -verify -ftrigraphs -DENABLED_TRIGRAPHS=1
// RUN: %clang_cc1 -std=c++1z %s -verify -fno-trigraphs -DENABLED_TRIGRAPHS=0

#ifdef __MVS__
#ifndef ENABLED_TRIGRAPHS
#define ENABLED_TRIGRAPHS 1
#endif
#endif

??= define foo ;

static_assert("??="[0] == '#', "");

// ??/
error here;

// Note, there is intentionally trailing whitespace one line below.
// ??/  
error here;

#if !ENABLED_TRIGRAPHS
// expected-error@11 {{}} expected-warning@11 {{trigraph ignored}}
// expected-error@13 {{failed}} expected-warning@13 {{trigraph ignored}} expected-note@13 {{evaluates to ''?' == '#''}}
// expected-error@16 {{}}
// expected-error@20 {{}}
#else
// expected-warning@11 {{trigraph converted}}
// expected-warning@13 {{trigraph converted}}
// expected-warning@19 {{backslash and newline separated by space}}
#endif
