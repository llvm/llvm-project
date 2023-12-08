// Test this without pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DSET
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DRESET
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DPUSH
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DPUSH_POP
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -include %s -verify -fsyntax-only -Wno-pragma-pack -DPUSH_POP_LABEL

// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DSET -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DSET -verify -include-pch %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DRESET -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DRESET -verify -include-pch %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH -verify -include-pch %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH_POP -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH_POP -verify -include-pch %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH_POP_LABEL -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -Wno-pragma-pack -DPUSH_POP_LABEL -verify -include-pch %t

#ifndef HEADER
#define HEADER

#ifdef SET
#pragma pack(1)
#endif

#ifdef RESET
#pragma pack(2)
#pragma pack ()
#endif

#ifdef PUSH
#pragma pack(1)
#pragma pack (push, 2)
#endif

#ifdef PUSH_POP
#pragma pack (push, 4)
#pragma pack (push, 2)
#pragma pack (pop)
#endif

#ifdef PUSH_POP_LABEL
#pragma pack (push, a, 4)
#pragma pack (push, b, 1)
#pragma pack (push, c, 2)
#pragma pack (pop, b)
#endif

#else

#ifdef SET
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack(pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#endif

#ifdef RESET
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma ()
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#endif

#ifdef PUSH
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 2}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack ()
#pragma pack (show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#endif

#ifdef PUSH_POP
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(pop) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#endif

#ifdef PUSH_POP_LABEL
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack(pop, c)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 4}}
#pragma pack(pop, a)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(pop)  // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#pragma pack(pop, b) // expected-warning {{#pragma pack(pop, ...) failed: stack empty}}
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#endif

#endif
