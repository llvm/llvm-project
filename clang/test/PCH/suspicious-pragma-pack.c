// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -verify -emit-pch -o %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -verify -include-pch %t

#ifndef HEADER
#define HEADER
#pragma pack (push, 1)
#endif
// expected-warning@-2 {{unterminated '#pragma pack (push, ...)' at end of file}}
