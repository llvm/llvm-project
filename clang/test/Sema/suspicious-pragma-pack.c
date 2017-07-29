// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DSAFE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DSAFE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DPUSH_SET_HERE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DRESET_HERE -DSAFE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DSET_FIRST_HEADER -DWARN_MODIFIED_HEADER -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DRESET_HERE -DSET_FIRST_HEADER -DWARN_MODIFIED_HEADER -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DPUSH_SET_HERE -DSET_FIRST_HEADER -DWARN_MODIFIED_HEADER -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DPUSH_SET_HERE -DSET_SECOND_HEADER -DWARN_MODIFIED_HEADER -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_HERE -DPUSH_SET_HERE -DSET_FIRST_HEADER -DSET_SECOND_HEADER -DWARN_MODIFIED_HEADER -verify %s

// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_POP_FIRST_HEADER -DSAFE -verify %s

// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -I %S/Inputs -DPUSH_SET_HERE -DNO_RECORD_1 -DNO_RECORD_2 -DSAFE -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack-suspicious-include -I %S/Inputs -DPUSH_SET_HERE -DNO_RECORD_1 -verify %s
// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -Wpragma-pack -Wno-pragma-pack-suspicious-include -I %S/Inputs -DPUSH_SET_HERE -DNO_RECORD_1 -DSAFE -verify %s

#ifdef SAFE
// expected-no-diagnostics
#endif

#ifdef PUSH_HERE
#pragma pack (push)
#endif

#ifdef PUSH_SET_HERE
#pragma pack (push, 4)
#ifndef SAFE
// expected-note@-2 {{previous '#pragma pack' directive that modifies alignment is here}}
// expected-warning@+9 {{non-default #pragma pack value changes the alignment of struct or union members in the included file}}
#endif
#endif

#ifdef RESET_HERE
#pragma pack (4)
#pragma pack () // no warning after reset as the value is default.
#endif

#include "pragma-pack1.h"

#ifdef WARN_MODIFIED_HEADER
// expected-warning@-3 {{the current #pragma pack aligment value is modified in the included file}}
#endif

#ifdef PUSH_SET_HERE
#pragma pack (pop)
#endif

#ifdef PUSH_HERE
#pragma pack (pop)
#endif
