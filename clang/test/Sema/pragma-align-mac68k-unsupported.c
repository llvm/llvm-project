// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -fsyntax-only -verify %s

/* expected-error {{mac68k alignment pragma is not supported}} */ #pragma options align=mac68k
