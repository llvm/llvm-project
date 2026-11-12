// Test this without pch.
// RUN: %clang_cc1 -D__DATE__= -D__TIMESTAMP__= -include %s -Wno-builtin-macro-redefined -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -D__DATE__= -D__TIMESTAMP__= -Wno-builtin-macro-redefined -emit-pch -o %t %s
// RUN: %clang_cc1 -D__DATE__= -D__TIMESTAMP__= -Wno-builtin-macro-redefined -include-pch %t -fsyntax-only -verify %s 

#if !defined(HEADER)
#define HEADER

#define __TIME__

#undef __TIMESTAMP__
#define __TIMESTAMP__

// FIXME: undefs don't work well with pchs yet, see PR31311
// Once that's fixed, add -U__COUNTER__ to all command lines and check that
// an attempt to use __COUNTER__ at the bottom produces an error in both non-pch
// and pch case (works fine in the former case already).
// Same for #undef __FILE__ right here and a use of that at the bottom.
//#undef __FILE__

// Also spot-check a predefine
#undef __STDC_HOSTED__

#else

const char s[] = __DATE__ " " __TIME__ " " __TIMESTAMP__;

// Check that we pick up __DATE__ from the -D flag:
int i = __DATE__ 4;

const int d = __STDC_HOSTED__; // expected-error{{use of undeclared identifier '__STDC_HOSTED__'}}

#endif
