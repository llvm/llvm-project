// RUN: %clang_cc1 -triple i386-apple-darwin9 -Wno-pragma-pack -fsyntax-only -verify %s
// expected-no-diagnostics

class C {
#pragma options align=natural
};
