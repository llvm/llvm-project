// Test this without pch.
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DSET
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DRESET
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DPUSH
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only \
// RUN:     -Wno-pragma-pack -DPUSH_POP
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -include %s -fsyntax-only -fdump-record-layouts \
// RUN:     -Wno-pragma-pack -DALIGN_NATURAL | \
// RUN:   FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DSET -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DSET -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DRESET -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DRESET -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH_POP -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH_POP -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DALIGN_NATURAL -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DALIGN_NATURAL \
// RUN:     -fdump-record-layouts -include-pch %t | \
// RUN:   FileCheck %s

// Test this without pch.
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DSET
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DRESET
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DPUSH
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -include %s -verify -fsyntax-only \
// RUN:     -Wno-pragma-pack -DPUSH_POP
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -include %s -fsyntax-only -fdump-record-layouts \
// RUN:     -Wno-pragma-pack -DALIGN_NATURAL

// Test with pch.
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DSET -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DSET -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DRESET -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DRESET -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH_POP -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DPUSH_POP -verify -include-pch %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DALIGN_NATURAL -emit-pch -o %t
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -fxl-pragma-pack %s -Wno-pragma-pack -DALIGN_NATURAL \
// RUN:     -fdump-record-layouts -include-pch %t | \
// RUN:   FileCheck %s

#ifndef HEADER
#define HEADER

#ifdef SET
#pragma pack(1)
#endif

#ifdef RESET
#pragma pack(2)
#pragma pack()
#endif

#ifdef PUSH
#pragma pack(1)
#pragma pack(push, 2)
#endif

#ifdef PUSH_POP
#pragma pack(push, 4)
#pragma pack(push, 2)
#pragma pack(pop)
#endif

#ifdef ALIGN_NATURAL
#pragma align(natural)
#endif

#else

#ifdef SET
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack(pop)
#endif

#ifdef RESET
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma()
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#endif

#ifdef PUSH
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 2}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack()
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(pop)  // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#endif

#ifdef PUSH_POP
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(pop)  // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#endif

#ifdef ALIGN_NATURAL
struct D {
  int i;
  double d;
} d;

int s = sizeof(d);

// CHECK:      *** Dumping AST Record Layout
// CHECK:          0 | struct D
// CHECK:          0 |   int i
// CHECK:          8 |   double d
// CHECK:            | [sizeof=16, align=4, preferredalign=8]
#endif

#endif
