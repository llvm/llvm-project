// RUN: %clang_cc1 %s -fsyntax-only -isystem %S/Inputs -verify

#define Foo 1

#pragma clang scope push
#undef Foo
#pragma clang scope pop

#ifndef Foo
#error "Foo is still defined!"
#endif

#define Bar 1 // expected-note{{previous definition is here}}
#pragma clang scope push
#define Bar 2 // expected-warning{{'Bar' macro redefined}}
#pragma clang scope pop

#if Bar != 1
#error "Bar is set back to 1"
#endif

#pragma clang scope push
#include <SomeHeaderThatDefinesAwfulThings.h>
#pragma clang scope pop

#ifdef max
#error "Nobody should ever define max as a macro!"
#endif

#pragma clang scope pop // expected-warning{{pragma scope pop could not pop, no matching push}}

#pragma clang scope enter // expected-error{{expected 'push' or 'pop'}}

#pragma clang scope push pop // expected-warning{{extra tokens at end of #pragma directive}}

#pragma clang scope () // expected-warning{{expected 'push' or 'pop'}} expected-warning{{extra tokens at end of #pragma directive}}
