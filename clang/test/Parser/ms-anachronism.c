// RUN: %clang_cc1 -triple i686-windows-msvc -fms-extensions -fsyntax-only -verify %s

struct {} __cdecl s; // expected-warning {{'__cdecl' only applies to function types; type here is 'struct}}
