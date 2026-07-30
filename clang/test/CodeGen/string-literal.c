// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -x c++ -std=c++11 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-CXX11 %s
// RUN: %clang_cc1 -x c -std=c11 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-C11 %s

#include <stddef.h>

#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
#endif

int main(void) {
  // CHECK-C: private unnamed_addr constant [10 x i8] c"abc\00\00\00\00\00\00\00", align 1
  // CHECK-C11: private unnamed_addr constant [10 x i8] c"abc\00\00\00\00\00\00\00", align 1
  // CHECK-CXX11: private unnamed_addr constant [10 x i8] c"abc\00\00\00\00\00\00\00", align 1
  char a[10] = "abc";

  // This should convert to utf8.
  // CHECK-C: private unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  // CHECK-C11: private unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  // CHECK-CXX11: private unnamed_addr constant [10 x i8] c"\E1\84\A0\C8\A0\F4\82\80\B0\00", align 1
  char b[10] = "\u1120\u0220\U00102030";

  // CHECK-C: private unnamed_addr constant [3 x i32] [i32 65, i32 66, i32 0], align 4
  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 65, i32 66, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 65, i32 66, i32 0], align 4
  const wchar_t *foo = L"AB";

  // CHECK-C: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110027, i32 0], align 4
  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110027, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110027, i32 0], align 4
  const wchar_t *bar = L"\u1234\U0010F00B";

  // CHECK-C: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110028, i32 0], align 4
  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110028, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 4660, i32 1110028, i32 0], align 4
  const wchar_t *baz = L"\u1234" "\U0010F00C";

#if __cplusplus >= 201103L || __STDC_VERSION__ >= 201112L
  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 67, i32 68, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 67, i32 68, i32 0], align 4
  const char32_t *c = U"CD";

  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 4661, i32 1110028, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 4661, i32 1110028, i32 0], align 4
  const char32_t *d = U"\u1235\U0010F00C";

  // CHECK-C11: private unnamed_addr constant [3 x i32] [i32 4661, i32 1110027, i32 0], align 4
  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 4661, i32 1110027, i32 0], align 4
  const char32_t *o = "\u1235" U"\U0010F00B";

  // CHECK-C11: private unnamed_addr constant [3 x i16] [i16 69, i16 70, i16 0], align 2
  // CHECK-CXX11: private unnamed_addr constant [3 x i16] [i16 69, i16 70, i16 0], align 2
  const char16_t *e = u"EF";

  // This should convert to utf16.
  // CHECK-C11: private unnamed_addr constant [5 x i16] [i16 4384, i16 544, i16 -9272, i16 -9168, i16 0], align 2
  // CHECK-CXX11: private unnamed_addr constant [5 x i16] [i16 4384, i16 544, i16 -9272, i16 -9168, i16 0], align 2
  const char16_t *f = u"\u1120\u0220\U00102030";

  // This should convert to utf16.
  // CHECK-C11: private unnamed_addr constant [5 x i16] [i16 4384, i16 800, i16 -9272, i16 -9168, i16 0], align 2
  // CHECK-CXX11: private unnamed_addr constant [5 x i16] [i16 4384, i16 800, i16 -9272, i16 -9168, i16 0], align 2
  const char16_t *p = u"\u1120\u0320" "\U00102030";

  // CHECK-C11: private unnamed_addr constant [4 x i8] c"def\00", align 1
  // CHECK-CXX11: private unnamed_addr constant [4 x i8] c"def\00", align 1
  const char *g = u8"def";

#ifdef __cplusplus
  // CHECK-CXX11: private unnamed_addr constant [4 x i8] c"ghi\00", align 1
  const char *h = R"foo(ghi)foo";

  // CHECK-CXX11: private unnamed_addr constant [4 x i8] c"jkl\00", align 1
  const char *i = u8R"bar(jkl)bar";

  // CHECK-CXX11: private unnamed_addr constant [3 x i16] [i16 71, i16 72, i16 0], align 2
  const char16_t *j = uR"foo(GH)foo";

  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 73, i32 74, i32 0], align 4
  const char32_t *k = UR"bar(IJ)bar";

  // CHECK-CXX11: private unnamed_addr constant [3 x i32] [i32 75, i32 76, i32 0], align 4
  const wchar_t *l = LR"bar(KL)bar";

  // CHECK-CXX11: private unnamed_addr constant [9 x i8] c"abc\\ndef\00", align 1
  const char *m = R"(abc\ndef)";

  // CHECK-CXX11: private unnamed_addr constant [8 x i8] c"abc\0Adef\00", align 1
  const char *n = R"(abc
def)";

  // CHECK-CXX11: private unnamed_addr constant [11 x i8] c"abc\0Adefghi\00", align 1
  const char *q = R"(abc
def)" "ghi";

  // CHECK-CXX11: private unnamed_addr constant [13 x i8] c"abc\\\0A??=\0Adef\00", align 1
  const char *r = R\
"(abc\
??=
def)";

  // CHECK-CXX11: private unnamed_addr constant [13 x i8] c"def\\\0A??=\0Aabc\00", align 1
  const char *s = u8R\
"(def\
??=
abc)";

  // CHECK-CXX11: private unnamed_addr constant [13 x i16] [i16 97, i16 98, i16 99, i16 92, i16 10, i16 63, i16 63, i16 61, i16 10, i16 100, i16 101, i16 102, i16 0], align 2
  const char16_t *t = uR\
"(abc\
??=
def)";

  // CHECK-CXX11: private unnamed_addr constant [13 x i32] [i32 97, i32 98, i32 99, i32 92, i32 10, i32 63, i32 63, i32 61, i32 10, i32 100, i32 101, i32 102, i32 0], align 4
  const char32_t *u = UR\
"(abc\
??=
def)";

  // CHECK-CXX11: private unnamed_addr constant [13 x i32] [i32 100, i32 101, i32 102, i32 92, i32 10, i32 63, i32 63, i32 61, i32 10, i32 97, i32 98, i32 99, i32 0], align 4
  const wchar_t *v = LR\
"(def\
??=
abc)";

#endif
#endif
}
