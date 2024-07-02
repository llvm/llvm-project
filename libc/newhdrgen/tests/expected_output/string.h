//===-- C standard library header string.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_STRING_H
#define LLVM_LIBC_STRING_H

#include "__llvm-libc-common.h"

#include "llvm-libc-macros/null-macro.h"

#include <llvm-libc-types/size_t.h>

__BEGIN_C_DECLS

void *memccpy(void *__restrict, const void *__restrict, int, size_t);

void *memchr(const void *, int, size_t);

int memcmp(const void *, const void *, size_t);

void *memcpy(void *__restrict, const void *__restrict, size_t);

void *memmem(const void *, size_t, const void *, size_t);

void *memmove(void *, const void *, size_t);

void *mempcpy(void *__restrict, const void *__restrict, size_t);

void *memrchr(const void *, int, size_t);

void *memset(void *, int, size_t);

void *memset_explicit(void *, int, size_t);

char *stpcpy(char *__restrict, const char *__restrict);

char *stpncpy(char *__restrict, const char *__restrict, size_t);

char *strcasestr(const char *, const char *);

char *strcat(char *__restrict, const char *__restrict);

char *strchr(const char *, int);

char *strchrnul(const char *, int);

int strcmp(const char *, const char *);

int strcoll(const char *, const char *);

char *strcpy(char *__restrict, const char *__restrict);

size_t strcspn(const char *, const char *);

char *strdup(const char *);

char *strerror(int);

char *strerror_r(int, char *, size_t);

size_t strlcat(const char *__restrict, const char *__restrict, size_t);

size_t strlcpy(const char *__restrict, const char *__restrict, size_t);

size_t strlen(const char *);

char *strncat(char *, const char *, size_t);

int strncmp(const char *, const char *, size_t);

char *strncpy(char *__restrict, const char *__restrict, size_t);

char *strndup(const char *, size_t);

size_t strnlen(const char *, size_t);

char *strpbrk(const char *, const char *);

char *strrchr(const char *, int);

char *strsep(char **__restrict, const char *__restrict);

char *strsignal(int);

size_t strspn(const char *, const char *);

char *strstr(const char *, const char *);

char *strtok(char *__restrict, const char *__restrict);

char *strtok_r(char *__restrict, const char *__restrict, char **__restrict);

size_t strxfrm(char *__restrict, const char *__restrict, size_t);

__END_C_DECLS

#endif // LLVM_LIBC_STRING_H
