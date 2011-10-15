//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.


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
