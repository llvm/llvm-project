//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


// CONFIG  rdar://6339747 but wasn't

#include <stdio.h>

int (*funcptr)(long);

int (*(^b)(char))(long);

int main(int argc, char *argv[])  {
	// implicit is fine
	b = ^(char x) { return funcptr; };
	// explicit never parses
	b = ^int (*(char x))(long) { return funcptr; };
        printf("%s: Success\n", argv[0]);
        return 0;
}
