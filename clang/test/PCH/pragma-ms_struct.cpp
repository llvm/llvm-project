// Test this without pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-apple-darwin9 -fsyntax-only -include %s -verify -std=c++11

// Test with pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-apple-darwin9 -emit-pch -o %t -std=c++11
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple i386-apple-darwin9 -fsyntax-only -include-pch %t -verify -std=c++11

// The first run line creates a pch, and since at that point HEADER is not
// defined, the only thing contained in the pch is the pragma. The second line
// then includes that pch, so HEADER is defined and the actual code is compiled.
// The check then makes sure that the pragma is in effect in the file that
// includes the pch.

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
struct SOffH {
  short m : 9;
  int q : 12;
};

#pragma ms_struct on

struct SOnH {
  short m : 9;
  int q : 12;
};

#else

struct SOnC {
  short m : 9;
  int q : 12;
};

static_assert(sizeof(SOffH) == 4, "");
static_assert(sizeof(SOnH) == 8, "");
static_assert(sizeof(SOnC) == 8, "");

#endif
