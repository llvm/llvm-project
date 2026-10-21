//===--- stdio.h - Stub header for tests ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _STDIO_H_
#define _STDIO_H_

// A header intended to contain C standard input and output library
// declarations.

typedef struct structFILE {} FILE;
extern FILE *stderr;

int printf(const char *, ...);
int fprintf(FILE *, const char *, ...);

#define NULL (0)

#endif // _STDIO_H_

