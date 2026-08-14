// RUN: %clang_cc1 -Wdeprecated %s -fsyntax-only -verify -Wunknown-pragmas -Wextra-tokens

#pragma clang deprecated_header( // expected-error {{expected string literal in #pragma clang deprecated_header}}
#pragma clang deprecated_header() // expected-error {{expected string literal in #pragma clang deprecated_header}}
#pragma clang deprecated_header("" // expected-error {{expected )}}
#pragma clang deprecated_header something // expected-warning {{extra tokens at end of #pragma clang deprecated_header directive}}
#pragma clang deprecated_header("") something // expected-warning {{extra tokens at end of #pragma clang deprecated_header directive}}

#include "deprecated-header.h" // expected-warning {{header is deprecated}}
#include "deprecated-header.h" // expected-warning {{header is deprecated}}

#include "deprecated-header-msg.h" // expected-warning {{header is deprecated: This is a shitty header}}
#include "deprecated-header-msg.h" // expected-warning {{header is deprecated: This is a shitty header}}
