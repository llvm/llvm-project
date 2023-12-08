// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack-suspicious-include -I %S/Inputs -DALIGN_SET_HERE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack-suspicious-include -I %S/Inputs -DRECORD_ALIGN -verify %s

#ifdef ALIGN_SET_HERE
#pragma align = natural // expected-warning {{unterminated '#pragma pack (push, ...)' at end of file}}
// expected-warning@+9 {{the current #pragma pack alignment value is modified in the included file}}
#endif

#ifdef RECORD_ALIGN
#pragma align = mac68k
// expected-note@-1 {{previous '#pragma pack' directive that modifies alignment is here}}
// expected-warning@+3 {{non-default #pragma pack value changes the alignment of struct or union members in the included file}}
#endif

#include "pragma-align-pack1.h"

#ifdef RECORD_ALIGN
#pragma align = reset
#endif
