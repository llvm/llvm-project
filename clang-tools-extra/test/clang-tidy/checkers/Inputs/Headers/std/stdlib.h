//===--- stdlib.h - Stub header for tests------ -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STDLIB_H_
#define _STDLIB_H_

void abort(void);
void _Exit(int);
void quick_exit(int);

long strtol(const char *Str, char **End, int Base);
long long strtoll(const char *Str, char **End, int Base);
unsigned long strtoul(const char *Str, char **End, int Base);
unsigned long long strtoull(const char *Str, char **End, int Base);

double strtod(const char *Str, char **End);
float strtof(const char *Str, char **End);
long double strtold(const char *Str, char **End);

int atoi(const char *Str);
long atol(const char *Str);

#endif // _STDLIB_H_
