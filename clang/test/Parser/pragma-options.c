// RUN: %clang_cc1 -triple i386-apple-darwin9 -fsyntax-only -verify %s

/* expected-warning {{expected 'align' following '#pragma options'}} */ #pragma options
/* expected-warning {{expected '=' following '#pragma options align'}} */ #pragma options align
/* expected-warning {{expected identifier in '#pragma options'}} */ #pragma options align =
/* expected-warning {{invalid alignment option in '#pragma options align'}} */ #pragma options align = foo
/* expected-warning {{extra tokens at end of '#pragma options'}} */ #pragma options align = reset foo

#pragma options align=natural
#pragma options align=reset
#pragma options align=mac68k
#pragma options align=power

/* expected-warning {{expected '=' following '#pragma align'}} */ #pragma align
/* expected-warning {{expected identifier in '#pragma align'}} */ #pragma align =
/* expected-warning {{invalid alignment option in '#pragma align'}} */ #pragma align = foo
/* expected-warning {{extra tokens at end of '#pragma align'}} */ #pragma align = reset foo

#pragma align=natural
#pragma align=reset
#pragma align=mac68k
#pragma align=power
