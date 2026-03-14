// RUN: %clang_cc1 -Wpragma-pack-suspicious-include -triple i686-apple-darwin9 -fsyntax-only -I%S/Inputs -verify %s

#pragma pack (push, 1) // expected-note {{previous '#pragma pack' directive that modifies alignment is here}}
#import "empty.h" // expected-warning {{non-default #pragma pack value changes the alignment of struct or union members in the included file}}

#pragma pack (pop)
