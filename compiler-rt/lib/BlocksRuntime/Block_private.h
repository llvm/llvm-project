// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//


#ifndef _BLOCK_PRIVATE_H_
#define _BLOCK_PRIVATE_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "Block.h"

#if __cplusplus
extern "C" {
#endif


// Values for Block_layout->flags to describe block objects
enum {
    BLOCK_DEALLOCATING =      (0x0001),  // runtime
    BLOCK_REFCOUNT_MASK =     (0xfffe),  // runtime
    BLOCK_NEEDS_FREE =        (1 << 24), // runtime
    BLOCK_HAS_COPY_DISPOSE =  (1 << 25), // compiler
    BLOCK_HAS_CTOR =          (1 << 26), // compiler: helpers have C++ code
    BLOCK_IS_GC =             (1 << 27), // runtime
    BLOCK_IS_GLOBAL =         (1 << 28), // compiler
    BLOCK_USE_STRET =         (1 << 29), // compiler: undefined if !BLOCK_HAS_SIGNATURE
    BLOCK_HAS_SIGNATURE  =    (1 << 30), // compiler
    BLOCK_HAS_EXTENDED_LAYOUT=(1 << 31)  // compiler
};

#define BLOCK_DESCRIPTOR_1 1
struct Block_descriptor_1 {
    unsigned long int reserved;
    unsigned long int size;
};

#define BLOCK_DESCRIPTOR_2 1
struct Block_descriptor_2 {
    // requires BLOCK_HAS_COPY_DISPOSE
    void (*copy)(void *dst, const void *src);
    void (*dispose)(const void *);
};

#define BLOCK_DESCRIPTOR_3 1
struct Block_descriptor_3 {
    // requires BLOCK_HAS_SIGNATURE
    const char *signature;
    const char *layout;     // contents depend on BLOCK_HAS_EXTENDED_LAYOUT
};

struct Block_layout {
    void *isa;
    volatile int32_t flags; // contains ref count
    int32_t reserved; 
    void (*invoke)(void *, ...);
    struct Block_descriptor_1 *descriptor;
    // imported variables
};


// Values for Block_byref->flags to describe __block variables
enum {
    // Byref refcount must use the same bits as Block_layout's refcount.
    // BLOCK_DEALLOCATING =      (0x0001),  // runtime
    // BLOCK_REFCOUNT_MASK =     (0xfffe),  // runtime

    BLOCK_BYREF_LAYOUT_MASK =       (0xf << 28), // compiler
    BLOCK_BYREF_LAYOUT_EXTENDED =   (  1 << 28), // compiler
    BLOCK_BYREF_LAYOUT_NON_OBJECT = (  2 << 28), // compiler
    BLOCK_BYREF_LAYOUT_STRONG =     (  3 << 28), // compiler
    BLOCK_BYREF_LAYOUT_WEAK =       (  4 << 28), // compiler
    BLOCK_BYREF_LAYOUT_UNRETAINED = (  5 << 28), // compiler

    BLOCK_BYREF_IS_GC =             (  1 << 27), // runtime

    BLOCK_BYREF_HAS_COPY_DISPOSE =  (  1 << 25), // compiler
    BLOCK_BYREF_NEEDS_FREE =        (  1 << 24), // runtime
};

struct Block_byref {
    void *isa;
    struct Block_byref *forwarding;
    volatile int32_t flags; // contains ref count
    uint32_t size;
};

struct Block_byref_2 {
    // requires BLOCK_BYREF_HAS_COPY_DISPOSE
    void (*byref_keep)(struct Block_byref *dst, struct Block_byref *src);
    void (*byref_destroy)(struct Block_byref *);
};

struct Block_byref_3 {
    // requires BLOCK_BYREF_LAYOUT_EXTENDED
    const char *layout;
};


// Extended layout encoding.

// Values for Block_descriptor_3->layout with BLOCK_HAS_EXTENDED_LAYOUT
// and for Block_byref_3->layout with BLOCK_BYREF_LAYOUT_EXTENDED

// If the layout field is less than 0x1000, then it is a compact encoding 
// of the form 0xXYZ: X strong pointers, then Y byref pointers, 
// then Z weak pointers.

// If the layout field is 0x1000 or greater, it points to a 
// string of layout bytes. Each byte is of the form 0xPN.
// Operator P is from the list below. Value N is a parameter for the operator.
// Byte 0x00 terminates the layout; remaining block data is non-pointer bytes.

enum {
    BLOCK_LAYOUT_ESCAPE = 0, // N=0 halt, rest is non-pointer. N!=0 reserved.
    BLOCK_LAYOUT_NON_OBJECT_BYTES = 1,    // N bytes non-objects
    BLOCK_LAYOUT_NON_OBJECT_WORDS = 2,    // N words non-objects
    BLOCK_LAYOUT_STRONG           = 3,    // N words strong pointers
    BLOCK_LAYOUT_BYREF            = 4,    // N words byref pointers
    BLOCK_LAYOUT_WEAK             = 5,    // N words weak pointers
    BLOCK_LAYOUT_UNRETAINED       = 6,    // N words unretained pointers
    BLOCK_LAYOUT_UNKNOWN_WORDS_7  = 7,    // N words, reserved
    BLOCK_LAYOUT_UNKNOWN_WORDS_8  = 8,    // N words, reserved
    BLOCK_LAYOUT_UNKNOWN_WORDS_9  = 9,    // N words, reserved
    BLOCK_LAYOUT_UNKNOWN_WORDS_A  = 0xA,  // N words, reserved
    BLOCK_LAYOUT_UNUSED_B         = 0xB,  // unspecified, reserved
    BLOCK_LAYOUT_UNUSED_C         = 0xC,  // unspecified, reserved
    BLOCK_LAYOUT_UNUSED_D         = 0xD,  // unspecified, reserved
    BLOCK_LAYOUT_UNUSED_E         = 0xE,  // unspecified, reserved
    BLOCK_LAYOUT_UNUSED_F         = 0xF,  // unspecified, reserved
};


// Runtime support functions used by compiler when generating copy/dispose helpers

// Values for _Block_object_assign() and _Block_object_dispose() parameters
enum {
    // see function implementation for a more complete description of these fields and combinations
    BLOCK_FIELD_IS_OBJECT   =  3,  // id, NSObject, __attribute__((NSObject)), block, ...
    BLOCK_FIELD_IS_BLOCK    =  7,  // a block variable
    BLOCK_FIELD_IS_BYREF    =  8,  // the on stack structure holding the __block variable
    BLOCK_FIELD_IS_WEAK     = 16,  // declared __weak, only used in byref copy helpers
    BLOCK_BYREF_CALLER      = 128, // called from __block (byref) copy/dispose support routines.
};

enum {
    BLOCK_ALL_COPY_DISPOSE_FLAGS = 
        BLOCK_FIELD_IS_OBJECT | BLOCK_FIELD_IS_BLOCK | BLOCK_FIELD_IS_BYREF |
        BLOCK_FIELD_IS_WEAK | BLOCK_BYREF_CALLER
};

// Runtime entry point called by compiler when assigning objects inside copy helper routines
BLOCK_EXPORT void _Block_object_assign(void *destAddr, const void *object, const int flags);
    // BLOCK_FIELD_IS_BYREF is only used from within block copy helpers


// runtime entry point called by the compiler when disposing of objects inside dispose helper routine
BLOCK_EXPORT void _Block_object_dispose(const void *object, const int flags);


// Other support functions

// runtime entry to get total size of a closure
BLOCK_EXPORT size_t Block_size(void *aBlock);

// indicates whether block was compiled with compiler that sets the ABI related metadata bits
BLOCK_EXPORT bool _Block_has_signature(void *aBlock);

// returns TRUE if return value of block is on the stack, FALSE otherwise
BLOCK_EXPORT bool _Block_use_stret(void *aBlock);

// Returns a string describing the block's parameter and return types.
// The encoding scheme is the same as Objective-C @encode.
// Returns NULL for blocks compiled with some compilers.
BLOCK_EXPORT const char * _Block_signature(void *aBlock);

// Returns a string describing the block's GC layout.
// This uses the GC skip/scan encoding.
// May return NULL.
BLOCK_EXPORT const char * _Block_layout(void *aBlock);

// Returns a string describing the block's layout.
// This uses the "extended layout" form described above.
// May return NULL.
BLOCK_EXPORT const char * _Block_extended_layout(void *aBlock);

// Callable only from the ARR weak subsystem while in exclusion zone
BLOCK_EXPORT bool _Block_tryRetain(const void *aBlock);

// Callable only from the ARR weak subsystem while in exclusion zone
BLOCK_EXPORT bool _Block_isDeallocating(const void *aBlock);


// the raw data space for runtime classes for blocks
// class+meta used for stack, malloc, and collectable based blocks
BLOCK_EXPORT void * _NSConcreteMallocBlock[32];
BLOCK_EXPORT void * _NSConcreteAutoBlock[32];
BLOCK_EXPORT void * _NSConcreteFinalizingBlock[32];
BLOCK_EXPORT void * _NSConcreteWeakBlockVariable[32];
// declared in Block.h
// BLOCK_EXPORT void * _NSConcreteGlobalBlock[32];
// BLOCK_EXPORT void * _NSConcreteStackBlock[32];


// the intercept routines that must be used under GC
BLOCK_EXPORT void _Block_use_GC( void *(*alloc)(const unsigned long, const bool isOne, const bool isObject),
                                  void (*setHasRefcount)(const void *, const bool),
                                  void (*gc_assign_strong)(void *, void **),
                                  void (*gc_assign_weak)(const void *, void *),
                                  void (*gc_memmove)(void *, void *, unsigned long));

// earlier version, now simply transitional
BLOCK_EXPORT void _Block_use_GC5( void *(*alloc)(const unsigned long, const bool isOne, const bool isObject),
                                  void (*setHasRefcount)(const void *, const bool),
                                  void (*gc_assign_strong)(void *, void **),
                                  void (*gc_assign_weak)(const void *, void *));

BLOCK_EXPORT void _Block_use_RR( void (*retain)(const void *),
                                 void (*release)(const void *));

struct Block_callbacks_RR {
    size_t  size;                   // size == sizeof(struct Block_callbacks_RR)
    void  (*retain)(const void *);
    void  (*release)(const void *);
    void  (*destructInstance)(const void *);
};
typedef struct Block_callbacks_RR Block_callbacks_RR;

BLOCK_EXPORT void _Block_use_RR2(const Block_callbacks_RR *callbacks);

// make a collectable GC heap based Block.  Not useful under non-GC.
BLOCK_EXPORT void *_Block_copy_collectable(const void *aBlock);

// thread-unsafe diagnostic
BLOCK_EXPORT const char *_Block_dump(const void *block);


// Obsolete

// first layout
struct Block_basic {
    void *isa;
    int Block_flags;  // int32_t
    int Block_size; // XXX should be packed into Block_flags
    void (*Block_invoke)(void *);
    void (*Block_copy)(void *dst, void *src);  // iff BLOCK_HAS_COPY_DISPOSE
    void (*Block_dispose)(void *);             // iff BLOCK_HAS_COPY_DISPOSE
    //long params[0];  // where const imports, __block storage references, etc. get laid down
} __attribute__((deprecated));


#if __cplusplus
}
#endif


#endif
