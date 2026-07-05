//===- not-null-terminated-result-cxx.h - Helper header ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header helps to maintain every function call checked by the
//  NotNullTerminatedResult checker.
//
//===----------------------------------------------------------------------===//

#pragma clang system_header

#include "not-null-terminated-result-c.h"

namespace std {
template <typename T>
struct basic_string {
  basic_string();
  const T *data() const;
  unsigned long size() const;
  unsigned long length() const;
};
typedef basic_string<char> string;
} // namespace std

size_t wcslen(const wchar_t *str);

template <size_t size>
char *strcpy(char (&dest)[size], const char *src);
template <size_t size>
wchar_t *wcscpy(wchar_t (&dest)[size], const wchar_t *src);
wchar_t *wcscpy(wchar_t *dest, const wchar_t *src);

template <size_t size>
errno_t strcpy_s(char (&dest)[size], const char *src);
template <size_t size>
errno_t wcscpy_s(wchar_t (&dest)[size], const wchar_t *src);
errno_t wcscpy_s(wchar_t *dest, size_t destSize, const wchar_t *src);

template <size_t size>
char *strncpy(char (&dest)[size], const char *src, size_t count);
template <size_t size>
wchar_t *wcsncpy(wchar_t (&dest)[size], const wchar_t *src, size_t count);
wchar_t *wcsncpy(wchar_t *dest, const wchar_t *src, size_t count);

template <size_t size>
errno_t strncpy_s(char (&dest)[size], const char *src, size_t count);
template <size_t size>
errno_t wcsncpy_s(wchar_t (&dest)[size], const wchar_t *src, size_t length);
errno_t wcsncpy_s(wchar_t *dest, size_t destSize, const wchar_t *src, size_t c);

wchar_t *wmemcpy(wchar_t *dest, const wchar_t *src, size_t count);
errno_t wmemcpy_s(wchar_t *dest, size_t destSize, const wchar_t *src, size_t c);

wchar_t *wcschr(const wchar_t *str, int c);
int wcsncmp(const wchar_t *str1, const wchar_t *str2, size_t count);
size_t wcsxfrm(wchar_t *dest, const wchar_t *src, size_t count);

void *wmemchr(const void *buffer, int c, size_t count);
void *wmemmove(void *dest, const void *src, size_t count);
errno_t wmemmove_s(void *dest, size_t destSize, const void *src, size_t count);
void *wmemset(void *dest, int c, size_t count);
