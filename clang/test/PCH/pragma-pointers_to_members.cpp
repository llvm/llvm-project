// Test this without pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-pc-win32 -fms-extensions -fsyntax-only -include %s -verify -std=c++11

// Test with pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-pc-win32  -fms-extensions -emit-pch -o %t -std=c++11
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-pc-win32  -fms-extensions -fsyntax-only -include-pch %t -verify -std=c++11

// The first run line creates a pch, and since at that point HEADER is not
// defined, the only thing contained in the pch is the pragma. The second line
// then includes that pch, so HEADER is defined and the actual code is compiled.
// The check then makes sure that the pragma is in effect in the file that
// includes the pch.

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct S0;
static_assert(sizeof(int S0::*) == 12, "");

struct S1;
struct S2;

#pragma pointers_to_members(full_generality, single_inheritance)

static_assert(sizeof(int S1::*) == 4, "");

#else

static_assert(sizeof(int S2::*) == 4, "");
static_assert(sizeof(int S0::*) == 12, "");

#endif
