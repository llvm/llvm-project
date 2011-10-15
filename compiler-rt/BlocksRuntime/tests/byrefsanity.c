//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG


#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <Block.h>

int
main(int argc, char *argv[])
{
    __block int var = 0;
    void (^b)(void) = ^{ var++; };
    
    //sanity(b);
    b();
    printf("%s: success!\n", argv[0]);
    return 0;
}


#if 1
/* replicated internal data structures: BEWARE, MAY CHANGE!!! */

enum {
    BLOCK_REFCOUNT_MASK =     (0xffff),
    BLOCK_NEEDS_FREE =        (1 << 24),
    BLOCK_HAS_COPY_DISPOSE =  (1 << 25),
    BLOCK_NO_COPY =           (1 << 26), // interim byref: no copies allowed
    BLOCK_IS_GC =             (1 << 27),
    BLOCK_IS_GLOBAL =         (1 << 28),
};

struct byref_id {
    struct byref_id *forwarding;
    int flags;//refcount;
    int size;
    void (*byref_keep)(struct byref_id *dst, struct byref_id *src);
    void (*byref_destroy)(struct byref_id *);
    int var;
};
struct Block_basic2 {
    void *isa;
    int Block_flags;  // int32_t
    int Block_size; // XXX should be packed into Block_flags
    void (*Block_invoke)(void *);
    void (*Block_copy)(void *dst, void *src);
    void (*Block_dispose)(void *);
    struct byref_id *ref;
};

void sanity(void *arg) {
    struct Block_basic2 *bb = (struct Block_basic2 *)arg;
    if ( ! (bb->Block_flags & BLOCK_HAS_COPY_DISPOSE)) {
        printf("missing copy/dispose helpers for byref data\n");
        exit(1);
    }
    struct byref_id *ref = bb->ref;
    if (ref->forwarding != ref) {
        printf("forwarding pointer should be %p but is %p\n", ref, ref->forwarding);
        exit(1);
    }
}
#endif


        
