// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -verify -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -verify -include-pch %t

#ifndef HEADER
#define HEADER
#pragma pack (push, 1)
#else
#pragma pack (2)
#endif
// expected-warning@-4 {{unterminated '#pragma pack (push, ...)' at end of file}}
