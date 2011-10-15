//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  goto.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/17/08.
 *
 */
 
// CONFIG rdar://6289031

#include <stdio.h>

int main(int argc, char *argv[])
{
	__block int val = 0;

	^{ val = 1; }();

	if (val == 0) {
		goto out_bad; // error: local byref variable val is in the scope of this goto
	}

        printf("%s: Success!\n", argv[0]);
	return 0;
out_bad:
        printf("%s: val not updated!\n", argv[0]);
	return 1;
}
