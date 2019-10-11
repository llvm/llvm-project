//===---------------- C standard library header ctype.h -----------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_CTYPE_H
#define LLVM_LIBC_CTYPE_H

#include <__llvm-libc-common.h>

__BEGIN_C_DECLS

int isalnum(int);

int isalpha(int);

int isblank(int);

int iscntrl(int);

int isdigit(int);

int isgraph(int);

int islower(int);

int isprint(int);

int ispunct(int);

int isspace(int);

int isupper(int);

int isxdigit(int);

int tolower(int);

int toupper(int);

__END_C_DECLS

#endif // LLVM_LIBC_CTYPE_H
