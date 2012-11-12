// RUN: %clang_cc1 -std=c++11 -ast-dump %s -triple x86_64-linux-gnu | FileCheck %s 

char c8[] = u8"test\0\\\"\t\a\b\234";
// CHECK: char c8[12] = (StringLiteral {{.*}} lvalue u8"test\000\\\"\t\a\b\234")

char16_t c16[] = u"test\0\\\"\t\a\b\234\u1234";
// CHECK: char16_t c16[13] = (StringLiteral {{.*}} lvalue u"test\000\\\"\t\a\b\234\u1234")

char32_t c32[] = U"test\0\\\"\t\a\b\234\u1234\U0010ffff"; // \
// CHECK: char32_t c32[14] = (StringLiteral {{.*}} lvalue U"test\000\\\"\t\a\b\234\u1234\U0010FFFF")

wchar_t wc[] = L"test\0\\\"\t\a\b\234\u1234\xffffffff"; // \
// CHECK: wchar_t wc[14] = (StringLiteral {{.*}} lvalue L"test\000\\\"\t\a\b\234\x1234\xFFFFFFFF")
