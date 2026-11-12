// RUN: %clang_cc1 -pedantic-errors -Wno-c11-extensions -std=c++11 -triple x86_64-linux-gnu -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic-errors -Wno-c11-extensions -std=c++11 -triple x86_64-linux-gnu -include-pch %t -verify %s
// REQUIRES: x86-registered-target
#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
extern thread_local int a;
extern _Thread_local int b;
extern int c;

#else

_Thread_local int a; // expected-error {{thread-local declaration of 'a' with static initialization follows declaration with dynamic initialization}}
// expected-note@7 {{previous declaration is here}}
thread_local int b; // expected-error {{thread-local declaration of 'b' with dynamic initialization follows declaration with static initialization}}
// expected-note@8 {{previous declaration is here}}
thread_local int c; // expected-error {{thread-local declaration of 'c' follows non-thread-local declaration}}
// expected-note@9 {{previous declaration is here}}

#endif
