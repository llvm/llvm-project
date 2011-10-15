//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

#include <stdio.h>

/* CONFIG rdar://6310599
 */

int main(int argc, char *argv[])
{
 	__block int flags;
 	__block void *isa;
 	
 	^{ flags=1; isa = (void *)isa; };
        printf("%s: success\n", argv[0]);
 	return 0;
}

