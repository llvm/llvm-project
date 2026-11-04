// RUN: %clang_cc1 -std=c23 %s -fsyntax-only -verify
// REQUIRES: system-windows


int null[] = {
#embed "NUL" limit(1) //expected-error {{device files are not yet supported by '#embed' directive}}
};
