//===---------------- C standard library header string.h ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_STRING_H
#define LLVM_LIBC_STRING_H

#include <__llvm-libc-common.h>

#define __need_size_t // To get only size_t from stddef.h
#define __need_NULL // To get only NULL from stddef.h
#include <stddef.h>

__BEGIN_C_DECLS

void *memcpy(void *__restrict, const void *__restrict, size_t);

void *memmove(void *, const void *, size_t);

int memcmp(const void *, const void *, size_t);

void *memchr(const void *, int, size_t);

void *memset(void *, int, size_t);

char *strcpy(char *__restrict, const char *__restrict);

char *strncpy(char *__restrict, const char *__restrict, size_t);

char *strcat(char *__restrict, const char *__restrict);

char *strncat(char *, const char *, size_t);

int strcmp(const char *, const char *);

int strcoll(const char *, const char *);

int strncmp(const char *, const char *, size_t);

size_t strxfrm(char *__restrict, const char *__restrict, size_t);

char *strchr(const char *, int);

size_t strcspn(const char *, const char *);

char *strpbrk(const char *, const char *);

char *strrchr(const char *, int c);

size_t strspn(const char *, const char *);

char *strstr(const char *, const char *);

char *strtok(char *__restrict, const char *__restrict);

char *strerror(int);

size_t strlen(const char *);

__END_C_DECLS

#endif // LLVM_LIBC_STRING_H
