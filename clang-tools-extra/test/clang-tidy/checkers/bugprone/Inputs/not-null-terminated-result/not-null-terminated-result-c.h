//===- not-null-terminated-result-c.h - Helper header -------------*- C -*-===//
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

typedef __typeof__(sizeof(int)) size_t;
typedef int errno_t;

size_t strlen(const char *str);
void *malloc(size_t size);
char *strerror(int errnum);
errno_t strerror_s(char *buffer, size_t bufferSize, int errnum);

char *strcpy(char *dest, const char *src);
errno_t strcpy_s(char *dest, size_t destSize, const char *src);
char *strncpy(char *dest, const char *src, size_t count);
errno_t strncpy_s(char *dest, size_t destSize, const char *src, size_t count);

void *memcpy(void *dest, const void *src, size_t count);
errno_t memcpy_s(void *dest, size_t destSize, const void *src, size_t count);

char *strchr(char *str, int c);
int strncmp(const char *str1, const char *str2, size_t count);
size_t strxfrm(char *dest, const char *src, size_t count);

void *memchr(const void *buffer, int c, size_t count);
void *memmove(void *dest, const void *src, size_t count);
errno_t memmove_s(void *dest, size_t destSize, const void *src, size_t count);
void *memset(void *dest, int c, size_t count);
