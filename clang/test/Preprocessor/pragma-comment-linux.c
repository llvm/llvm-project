// RUN: %clang_cc1 -triple i686-unknown-linux-gnu -fsyntax-only -verify %s -Wunknown-pragmas

#pragma comment(linker, "")
// expected-warning@-1 {{'#pragma comment linker' ignored}}

