//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG open rdar://6416474
// was  rdar://5847976
// was  rdar://6348320 

#include <stdio.h>
#include <Block.h>

int verbose = 0;

int main(int argc, char* argv[]) {
	
        if (argc > 1) verbose = 1;
        
	__block void (^recursive_local_block)(int);
		
	if (verbose) printf("recursive_local_block is a local recursive block\n");
	recursive_local_block = ^(int i) {
		if (verbose) printf("%d\n", i);
		if (i > 0) {
			recursive_local_block(i - 1);
		}
    };

	if (verbose) printf("recursive_local_block's address is %p, running it:\n", (void*)recursive_local_block);
	recursive_local_block(5);

	if (verbose) printf("Creating other_local_block: a local block that calls recursive_local_block\n");

	void (^other_local_block)(int) = ^(int i) {
		if (verbose) printf("other_local_block running\n");
		recursive_local_block(i);
    };

	if (verbose) printf("other_local_block's address is %p, running it:\n", (void*)other_local_block);
		
	other_local_block(5);
	
#if __APPLE_CC__ >= 5627
	if (verbose) printf("Creating other_copied_block: a Block_copy of a block that will call recursive_local_block\n");

	void (^other_copied_block)(int) = Block_copy(^(int i) {
		if (verbose) printf("other_copied_block running\n");
		recursive_local_block(i);
    });
		
	if (verbose) printf("other_copied_block's address is %p, running it:\n", (void*)other_copied_block);
	
	other_copied_block(5);
#endif

	__block void (^recursive_copy_block)(int);

	if (verbose) printf("Creating recursive_copy_block: a Block_copy of a block that will call recursive_copy_block recursively\n");

	recursive_copy_block = Block_copy(^(int i) {
		if (verbose) printf("%d\n", i);
		if (i > 0) {
			recursive_copy_block(i - 1);
		}
    });
		
	if (verbose) printf("recursive_copy_block's address is %p, running it:\n", (void*)recursive_copy_block);

	recursive_copy_block(5);
        
        printf("%s: Success\n", argv[0]);
	return 0;
}
