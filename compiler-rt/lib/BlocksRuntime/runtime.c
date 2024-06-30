// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//

#include "Block_private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if HAVE_OBJC
#define __USE_GNU
#include <dlfcn.h>
#endif
#if __has_include(<os/assumes.h>)
#include <os/assumes.h>
#else
#include <assert.h> 
#endif
#ifndef os_assumes
#define os_assumes(_x) _x
#endif
#ifndef os_assert
#define os_assert(_x) assert(_x)
#endif

#if !defined(__has_builtin)
#define __has_builtin(builtin) 0
#endif

#if __has_builtin(__sync_bool_compare_and_swap)
#define OSAtomicCompareAndSwapInt(_Old, _New, _Ptr)                            \
  __sync_bool_compare_and_swap(_Ptr, _Old, _New)
#else
#define _CRT_SECURE_NO_WARNINGS 1
#include <Windows.h>
static __inline bool OSAtomicCompareAndSwapInt(int oldi, int newi,
                                               int volatile *dst) {
  // fixme barrier is overkill -- see objc-os.h
  int original = InterlockedCompareExchange((LONG volatile *)dst, newi, oldi);
  return (original == oldi);
}
#endif

/***********************
Globals
************************/

#if HAVE_OBJC
static void *_Block_copy_class = _NSConcreteMallocBlock;
static void *_Block_copy_finalizing_class = _NSConcreteMallocBlock;
static int _Block_copy_flag = BLOCK_NEEDS_FREE;
#endif
static int _Byref_flag_initial_value = BLOCK_BYREF_NEEDS_FREE | 4;  // logical 2

static bool isGC = false;

/*******************************************************************************
Internal Utilities
********************************************************************************/


static int32_t latching_incr_int(volatile int32_t *where) {
    while (1) {
        int32_t old_value = *where;
        if ((old_value & BLOCK_REFCOUNT_MASK) == BLOCK_REFCOUNT_MASK) {
            return BLOCK_REFCOUNT_MASK;
        }
        if (OSAtomicCompareAndSwapInt(old_value, old_value+2, where)) {
            return old_value+2;
        }
    }
}

static bool latching_incr_int_not_deallocating(volatile int32_t *where) {
    while (1) {
        int32_t old_value = *where;
        if (old_value & BLOCK_DEALLOCATING) {
            // if deallocating we can't do this
            return false;
        }
        if ((old_value & BLOCK_REFCOUNT_MASK) == BLOCK_REFCOUNT_MASK) {
            // if latched, we're leaking this block, and we succeed
            return true;
        }
        if (OSAtomicCompareAndSwapInt(old_value, old_value+2, where)) {
            // otherwise, we must store a new retained value without the deallocating bit set
            return true;
        }
    }
}


// return should_deallocate?
static bool latching_decr_int_should_deallocate(volatile int32_t *where) {
    while (1) {
        int32_t old_value = *where;
        if ((old_value & BLOCK_REFCOUNT_MASK) == BLOCK_REFCOUNT_MASK) {
            return false; // latched high
        }
        if ((old_value & BLOCK_REFCOUNT_MASK) == 0) {
            return false;   // underflow, latch low
        }
        int32_t new_value = old_value - 2;
        bool result = false;
        if ((old_value & (BLOCK_REFCOUNT_MASK|BLOCK_DEALLOCATING)) == 2) {
            new_value = old_value - 1;
            result = true;
        }
        if (OSAtomicCompareAndSwapInt(old_value, new_value, where)) {
            return result;
        }
    }
}

// hit zero?
static bool latching_decr_int_now_zero(volatile int32_t *where) {
    while (1) {
        int32_t old_value = *where;
        if ((old_value & BLOCK_REFCOUNT_MASK) == BLOCK_REFCOUNT_MASK) {
            return false; // latched high
        }
        if ((old_value & BLOCK_REFCOUNT_MASK) == 0) {
            return false;   // underflow, latch low
        }
        int32_t new_value = old_value - 2;
        if (OSAtomicCompareAndSwapInt(old_value, new_value, where)) {
            return (new_value & BLOCK_REFCOUNT_MASK) == 0;
        }
    }
}


/***********************
GC support stub routines
************************/
#if !defined(_MSC_VER) || defined(__clang__)
#pragma mark GC Support Routines
#endif



static void *_Block_alloc_default(size_t size, const bool initialCountIsOne, const bool isObject) {
	(void)initialCountIsOne;
	(void)isObject;
    return malloc(size);
}

static void _Block_assign_default(void *value, void **destptr) {
    *destptr = value;
}

static void _Block_setHasRefcount_default(const void *ptr, const bool hasRefcount) {
	(void)ptr;
	(void)hasRefcount;
}

#if HAVE_OBJC
static void _Block_do_nothing(const void *aBlock) { }
#endif

static void _Block_retain_object_default(const void *ptr) {
	(void)ptr;
}

static void _Block_release_object_default(const void *ptr) {
	(void)ptr;
}

static void _Block_assign_weak_default(const void *ptr, void *dest) {
#if !defined(_WIN32)
    *(long *)dest = (long)ptr;
#else
    *(void **)dest = (void *)ptr;
#endif
}

static void _Block_memmove_default(void *dst, void *src, unsigned long size) {
    memmove(dst, src, (size_t)size);
}

#if HAVE_OBJC
static void _Block_memmove_gc_broken(void *dest, void *src, unsigned long size) {
    void **destp = (void **)dest;
    void **srcp = (void **)src;
    while (size) {
        _Block_assign_default(*srcp, destp);
        destp++;
        srcp++;
        size -= sizeof(void *);
    }
}
#endif

static void _Block_destructInstance_default(const void *aBlock) {
	(void)aBlock;
}

/**************************************************************************
GC support callout functions - initially set to stub routines
***************************************************************************/

static void *(*_Block_allocator)(size_t, const bool isOne, const bool isObject) = _Block_alloc_default;
static void (*_Block_deallocator)(const void *) = (void (*)(const void *))free;
static void (*_Block_assign)(void *value, void **destptr) = _Block_assign_default;
static void (*_Block_setHasRefcount)(const void *ptr, const bool hasRefcount) = _Block_setHasRefcount_default;
static void (*_Block_retain_object)(const void *ptr) = _Block_retain_object_default;
static void (*_Block_release_object)(const void *ptr) = _Block_release_object_default;
static void (*_Block_assign_weak)(const void *dest, void *ptr) = _Block_assign_weak_default;
static void (*_Block_memmove)(void *dest, void *src, unsigned long size) = _Block_memmove_default;
static void (*_Block_destructInstance) (const void *aBlock) = _Block_destructInstance_default;


#if HAVE_OBJC
/**************************************************************************
GC support SPI functions - called from ObjC runtime and CoreFoundation
***************************************************************************/

// Public SPI
// Called from objc-auto to turn on GC.
// version 3, 4 arg, but changed 1st arg
void _Block_use_GC( void *(*alloc)(size_t, const bool isOne, const bool isObject),
                    void (*setHasRefcount)(const void *, const bool),
                    void (*gc_assign)(void *, void **),
                    void (*gc_assign_weak)(const void *, void *),
                    void (*gc_memmove)(void *, void *, unsigned long)) {

    isGC = true;
    _Block_allocator = alloc;
    _Block_deallocator = _Block_do_nothing;
    _Block_assign = gc_assign;
    _Block_copy_flag = BLOCK_IS_GC;
    _Block_copy_class = _NSConcreteAutoBlock;
    // blocks with ctors & dtors need to have the dtor run from a class with a finalizer
    _Block_copy_finalizing_class = _NSConcreteFinalizingBlock;
    _Block_setHasRefcount = setHasRefcount;
    _Byref_flag_initial_value = BLOCK_BYREF_IS_GC;   // no refcount
    _Block_retain_object = _Block_do_nothing;
    _Block_release_object = _Block_do_nothing;
    _Block_assign_weak = gc_assign_weak;
    _Block_memmove = gc_memmove;
}

// transitional
void _Block_use_GC5( void *(*alloc)(size_t, const bool isOne, const bool isObject),
                    void (*setHasRefcount)(const void *, const bool),
                    void (*gc_assign)(void *, void **),
                    void (*gc_assign_weak)(const void *, void *)) {
    // until objc calls _Block_use_GC it will call us; supply a broken internal memmove implementation until then
    _Block_use_GC(alloc, setHasRefcount, gc_assign, gc_assign_weak, _Block_memmove_gc_broken);
}

 
// Called from objc-auto to alternatively turn on retain/release.
// Prior to this the only "object" support we can provide is for those
// super special objects that live in libSystem, namely dispatch queues.
// Blocks and Block_byrefs have their own special entry points.
BLOCK_EXPORT
void _Block_use_RR( void (*retain)(const void *),
                    void (*release)(const void *)) {
    _Block_retain_object = retain;
    _Block_release_object = release;
    _Block_destructInstance = dlsym(RTLD_DEFAULT, "objc_destructInstance");
}
#endif // HAVE_OBJC

// Called from CF to indicate MRR. Newer version uses a versioned structure, so we can add more functions
// without defining a new entry point.
BLOCK_EXPORT
void _Block_use_RR2(const Block_callbacks_RR *callbacks) {
    _Block_retain_object = callbacks->retain;
    _Block_release_object = callbacks->release;
    _Block_destructInstance = callbacks->destructInstance;
}

/****************************************************************************
Accessors for block descriptor fields
*****************************************************************************/
#if 0
static struct Block_descriptor_1 * _Block_descriptor_1(struct Block_layout *aBlock)
{
    return aBlock->descriptor;
}
#endif

static struct Block_descriptor_2 * _Block_descriptor_2(struct Block_layout *aBlock)
{
    if (! (aBlock->flags & BLOCK_HAS_COPY_DISPOSE)) return NULL;
    uint8_t *desc = (uint8_t *)aBlock->descriptor;
    desc += sizeof(struct Block_descriptor_1);
    return (struct Block_descriptor_2 *)desc;
}

static struct Block_descriptor_3 * _Block_descriptor_3(struct Block_layout *aBlock)
{
    if (! (aBlock->flags & BLOCK_HAS_SIGNATURE)) return NULL;
    uint8_t *desc = (uint8_t *)aBlock->descriptor;
    desc += sizeof(struct Block_descriptor_1);
    if (aBlock->flags & BLOCK_HAS_COPY_DISPOSE) {
        desc += sizeof(struct Block_descriptor_2);
    }
    return (struct Block_descriptor_3 *)desc;
}

static __inline bool _Block_has_layout(struct Block_layout *aBlock) {
    if (! (aBlock->flags & BLOCK_HAS_SIGNATURE)) return false;
    uint8_t *desc = (uint8_t *)aBlock->descriptor;
    desc += sizeof(struct Block_descriptor_1);
    if (aBlock->flags & BLOCK_HAS_COPY_DISPOSE) {
        desc += sizeof(struct Block_descriptor_2);
    }
    return ((struct Block_descriptor_3 *)desc)->layout != NULL;
}    

static void _Block_call_copy_helper(void *result, struct Block_layout *aBlock)
{
    struct Block_descriptor_2 *desc = _Block_descriptor_2(aBlock);
    if (!desc) return;

    (*desc->copy)(result, aBlock); // do fixup
}

static void _Block_call_dispose_helper(struct Block_layout *aBlock)
{
    struct Block_descriptor_2 *desc = _Block_descriptor_2(aBlock);
    if (!desc) return;

    (*desc->dispose)(aBlock);
}

/*******************************************************************************
Internal Support routines for copying
********************************************************************************/

#if !defined(_MSC_VER) || defined(__clang__)
#pragma mark Copy/Release support
#endif

// Copy, or bump refcount, of a block.  If really copying, call the copy helper if present.
static void *_Block_copy_internal(const void *arg, const bool wantsOne) {
    struct Block_layout *aBlock;

    if (!arg) return NULL;
    
    
    // The following would be better done as a switch statement
    aBlock = (struct Block_layout *)arg;
    if (aBlock->flags & BLOCK_NEEDS_FREE) {
        // latches on high
        latching_incr_int(&aBlock->flags);
        return aBlock;
    }
    else if (aBlock->flags & BLOCK_IS_GC) {
        // GC refcounting is expensive so do most refcounting here.
        if (wantsOne && ((latching_incr_int(&aBlock->flags) & BLOCK_REFCOUNT_MASK) == 2)) {
            // Tell collector to hang on this - it will bump the GC refcount version
            _Block_setHasRefcount(aBlock, true);
        }
        return aBlock;
    }
    else if (aBlock->flags & BLOCK_IS_GLOBAL) {
        return aBlock;
    }

    // Its a stack block.  Make a copy.
    if (!isGC) {
        struct Block_layout *result = malloc(aBlock->descriptor->size);
        if (!result) return NULL;
        memmove(result, aBlock, aBlock->descriptor->size); // bitcopy first
        // reset refcount
        result->flags &= ~(BLOCK_REFCOUNT_MASK|BLOCK_DEALLOCATING);    // XXX not needed
        result->flags |= BLOCK_NEEDS_FREE | 2;  // logical refcount 1
        result->isa = _NSConcreteMallocBlock;
        _Block_call_copy_helper(result, aBlock);
        return result;
    }
    else {
        // Under GC want allocation with refcount 1 so we ask for "true" if wantsOne
        // This allows the copy helper routines to make non-refcounted block copies under GC
        int32_t flags = aBlock->flags;
        bool hasCTOR = (flags & BLOCK_HAS_CTOR) != 0;
        struct Block_layout *result = _Block_allocator(aBlock->descriptor->size, wantsOne, hasCTOR || _Block_has_layout(aBlock));
        if (!result) return NULL;
        memmove(result, aBlock, aBlock->descriptor->size); // bitcopy first
        // reset refcount
        // if we copy a malloc block to a GC block then we need to clear NEEDS_FREE.
        flags &= ~(BLOCK_NEEDS_FREE|BLOCK_REFCOUNT_MASK|BLOCK_DEALLOCATING);   // XXX not needed
        if (wantsOne)
            flags |= BLOCK_IS_GC | 2;
        else
            flags |= BLOCK_IS_GC;
        result->flags = flags;
        _Block_call_copy_helper(result, aBlock);
        if (hasCTOR) {
            result->isa = _NSConcreteFinalizingBlock;
        }
        else {
            result->isa = _NSConcreteAutoBlock;
        }
        return result;
    }
}





// Runtime entry points for maintaining the sharing knowledge of byref data blocks.

// A closure has been copied and its fixup routine is asking us to fix up the reference to the shared byref data
// Closures that aren't copied must still work, so everyone always accesses variables after dereferencing the forwarding ptr.
// We ask if the byref pointer that we know about has already been copied to the heap, and if so, increment it.
// Otherwise we need to copy it and update the stack forwarding pointer
static void _Block_byref_assign_copy(void *dest, const void *arg, const int flags) {
    struct Block_byref **destp = (struct Block_byref **)dest;
    struct Block_byref *src = (struct Block_byref *)arg;
        
    if (src->forwarding->flags & BLOCK_BYREF_IS_GC) {
        ;   // don't need to do any more work
    }
    else if ((src->forwarding->flags & BLOCK_REFCOUNT_MASK) == 0) {
        // src points to stack
        bool isWeak = ((flags & (BLOCK_FIELD_IS_BYREF|BLOCK_FIELD_IS_WEAK)) == (BLOCK_FIELD_IS_BYREF|BLOCK_FIELD_IS_WEAK));
        // if its weak ask for an object (only matters under GC)
        struct Block_byref *copy = (struct Block_byref *)_Block_allocator(src->size, false, isWeak);
        copy->flags = src->flags | _Byref_flag_initial_value; // non-GC one for caller, one for stack
        copy->forwarding = copy; // patch heap copy to point to itself (skip write-barrier)
        src->forwarding = copy;  // patch stack to point to heap copy
        copy->size = src->size;
        if (isWeak) {
            copy->isa = &_NSConcreteWeakBlockVariable;  // mark isa field so it gets weak scanning
        }
        if (src->flags & BLOCK_BYREF_HAS_COPY_DISPOSE) {
            // Trust copy helper to copy everything of interest
            // If more than one field shows up in a byref block this is wrong XXX
            struct Block_byref_2 *src2 = (struct Block_byref_2 *)(src+1);
            struct Block_byref_2 *copy2 = (struct Block_byref_2 *)(copy+1);
            copy2->byref_keep = src2->byref_keep;
            copy2->byref_destroy = src2->byref_destroy;

            if (src->flags & BLOCK_BYREF_LAYOUT_EXTENDED) {
                struct Block_byref_3 *src3 = (struct Block_byref_3 *)(src2+1);
                struct Block_byref_3 *copy3 = (struct Block_byref_3*)(copy2+1);
                copy3->layout = src3->layout;
            }

            (*src2->byref_keep)(copy, src);
        }
        else {
            // just bits.  Blast 'em using _Block_memmove in case they're __strong
            // This copy includes Block_byref_3, if any.
            _Block_memmove(copy+1, src+1,
                           src->size - sizeof(struct Block_byref));
        }
    }
    // already copied to heap
    else if ((src->forwarding->flags & BLOCK_BYREF_NEEDS_FREE) == BLOCK_BYREF_NEEDS_FREE) {
        latching_incr_int(&src->forwarding->flags);
    }
    // assign byref data block pointer into new Block
    _Block_assign(src->forwarding, (void **)destp);
}

// Old compiler SPI
static void _Block_byref_release(const void *arg) {
    struct Block_byref *byref = (struct Block_byref *)arg;

    // dereference the forwarding pointer since the compiler isn't doing this anymore (ever?)
    byref = byref->forwarding;

    // To support C++ destructors under GC we arrange for there to be a finalizer for this
    // by using an isa that directs the code to a finalizer that calls the byref_destroy method.
    if ((byref->flags & BLOCK_BYREF_NEEDS_FREE) == 0) {
        return; // stack or GC or global
    }
    os_assert(byref->flags & BLOCK_REFCOUNT_MASK);
    if (latching_decr_int_should_deallocate(&byref->flags)) {
        if (byref->flags & BLOCK_BYREF_HAS_COPY_DISPOSE) {
            struct Block_byref_2 *byref2 = (struct Block_byref_2 *)(byref+1);
            (*byref2->byref_destroy)(byref);
        }
        _Block_deallocator((struct Block_layout *)byref);
    }
}


/************************************************************
 *
 * API supporting SPI
 * _Block_copy, _Block_release, and (old) _Block_destroy
 *
 ***********************************************************/

#if !defined(_MSC_VER) || defined(__clang__)
#pragma mark SPI/API
#endif

BLOCK_EXPORT
void *_Block_copy(const void *arg) {
    return _Block_copy_internal(arg, true);
}


// API entry point to release a copied Block
BLOCK_EXPORT
void _Block_release(const void *arg) {
    struct Block_layout *aBlock = (struct Block_layout *)arg;
    if (!aBlock 
        || (aBlock->flags & BLOCK_IS_GLOBAL)
        || ((aBlock->flags & (BLOCK_IS_GC|BLOCK_NEEDS_FREE)) == 0)
        ) return;
    if (aBlock->flags & BLOCK_IS_GC) {
        if (latching_decr_int_now_zero(&aBlock->flags)) {
            // Tell GC we no longer have our own refcounts.  GC will decr its refcount
            // and unless someone has done a CFRetain or marked it uncollectable it will
            // now be subject to GC reclamation.
            _Block_setHasRefcount(aBlock, false);
        }
    }
    else if (aBlock->flags & BLOCK_NEEDS_FREE) {
        if (latching_decr_int_should_deallocate(&aBlock->flags)) {
            _Block_call_dispose_helper(aBlock);
            _Block_destructInstance(aBlock);
            _Block_deallocator(aBlock);
        }
    }
}

BLOCK_EXPORT
bool _Block_tryRetain(const void *arg) {
    struct Block_layout *aBlock = (struct Block_layout *)arg;
    return latching_incr_int_not_deallocating(&aBlock->flags);
}

BLOCK_EXPORT
bool _Block_isDeallocating(const void *arg) {
    struct Block_layout *aBlock = (struct Block_layout *)arg;
    return (aBlock->flags & BLOCK_DEALLOCATING) != 0;
}

// Old Compiler SPI point to release a copied Block used by the compiler in dispose helpers
static void _Block_destroy(const void *arg) {
    struct Block_layout *aBlock;
    if (!arg) return;
    aBlock = (struct Block_layout *)arg;
    if (aBlock->flags & BLOCK_IS_GC) {
        // assert(aBlock->Block_flags & BLOCK_HAS_CTOR);
        return; // ignore, we are being called because of a DTOR
    }
    _Block_release(aBlock);
}



/************************************************************
 *
 * SPI used by other layers
 *
 ***********************************************************/

// SPI, also internal.  Called from NSAutoBlock only under GC
BLOCK_EXPORT
void *_Block_copy_collectable(const void *aBlock) {
    return _Block_copy_internal(aBlock, false);
}


// SPI
BLOCK_EXPORT
size_t Block_size(void *aBlock) {
    return ((struct Block_layout *)aBlock)->descriptor->size;
}

BLOCK_EXPORT
bool _Block_use_stret(void *aBlock) {
    struct Block_layout *layout = (struct Block_layout *)aBlock;

    int requiredFlags = BLOCK_HAS_SIGNATURE | BLOCK_USE_STRET;
    return (layout->flags & requiredFlags) == requiredFlags;
}

// Checks for a valid signature, not merely the BLOCK_HAS_SIGNATURE bit.
BLOCK_EXPORT
bool _Block_has_signature(void *aBlock) {
    return _Block_signature(aBlock) ? true : false;
}

BLOCK_EXPORT
const char * _Block_signature(void *aBlock)
{
    struct Block_descriptor_3 *desc3 = _Block_descriptor_3(aBlock);
    if (!desc3) return NULL;

    return desc3->signature;
}

BLOCK_EXPORT
const char * _Block_layout(void *aBlock)
{
    // Don't return extended layout to callers expecting GC layout
    struct Block_layout *layout = (struct Block_layout *)aBlock;
    if (layout->flags & BLOCK_HAS_EXTENDED_LAYOUT) return NULL;

    struct Block_descriptor_3 *desc3 = _Block_descriptor_3(aBlock);
    if (!desc3) return NULL;

    return desc3->layout;
}

BLOCK_EXPORT
const char * _Block_extended_layout(void *aBlock)
{
    // Don't return GC layout to callers expecting extended layout
    struct Block_layout *layout = (struct Block_layout *)aBlock;
    if (! (layout->flags & BLOCK_HAS_EXTENDED_LAYOUT)) return NULL;

    struct Block_descriptor_3 *desc3 = _Block_descriptor_3(aBlock);
    if (!desc3) return NULL;

    // Return empty string (all non-object bytes) instead of NULL 
    // so callers can distinguish "empty layout" from "no layout".
    if (!desc3->layout) return "";
    else return desc3->layout;
}

#if !defined(_MSC_VER) || defined(__clang__)
#pragma mark Compiler SPI entry points
#endif

    
/*******************************************************

Entry points used by the compiler - the real API!


A Block can reference four different kinds of things that require help when the Block is copied to the heap.
1) C++ stack based objects
2) References to Objective-C objects
3) Other Blocks
4) __block variables

In these cases helper functions are synthesized by the compiler for use in Block_copy and Block_release, called the copy and dispose helpers.  The copy helper emits a call to the C++ const copy constructor for C++ stack based objects and for the rest calls into the runtime support function _Block_object_assign.  The dispose helper has a call to the C++ destructor for case 1 and a call into _Block_object_dispose for the rest.

The flags parameter of _Block_object_assign and _Block_object_dispose is set to
	* BLOCK_FIELD_IS_OBJECT (3), for the case of an Objective-C Object,
	* BLOCK_FIELD_IS_BLOCK (7), for the case of another Block, and
	* BLOCK_FIELD_IS_BYREF (8), for the case of a __block variable.
If the __block variable is marked weak the compiler also or's in BLOCK_FIELD_IS_WEAK (16)

So the Block copy/dispose helpers should only ever generate the four flag values of 3, 7, 8, and 24.

When  a __block variable is either a C++ object, an Objective-C object, or another Block then the compiler also generates copy/dispose helper functions.  Similarly to the Block copy helper, the "__block" copy helper (formerly and still a.k.a. "byref" copy helper) will do a C++ copy constructor (not a const one though!) and the dispose helper will do the destructor.  And similarly the helpers will call into the same two support functions with the same values for objects and Blocks with the additional BLOCK_BYREF_CALLER (128) bit of information supplied.

So the __block copy/dispose helpers will generate flag values of 3 or 7 for objects and Blocks respectively, with BLOCK_FIELD_IS_WEAK (16) or'ed as appropriate and always 128 or'd in, for the following set of possibilities:
	__block id                   128+3       (0x83)
	__block (^Block)             128+7       (0x87)
    __weak __block id            128+3+16    (0x93)
	__weak __block (^Block)      128+7+16    (0x97)
        

********************************************************/

//
// When Blocks or Block_byrefs hold objects then their copy routine helpers use this entry point
// to do the assignment.
//
BLOCK_EXPORT
void _Block_object_assign(void *destAddr, const void *object, const int flags) {
    switch (os_assumes(flags & BLOCK_ALL_COPY_DISPOSE_FLAGS)) {
      case BLOCK_FIELD_IS_OBJECT:
        /*******
        id object = ...;
        [^{ object; } copy];
        ********/
            
        _Block_retain_object(object);
        _Block_assign((void *)object, destAddr);
        break;

      case BLOCK_FIELD_IS_BLOCK:
        /*******
        void (^object)(void) = ...;
        [^{ object; } copy];
        ********/

        _Block_assign(_Block_copy_internal(object, false), destAddr);
        break;
    
      case BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_WEAK:
      case BLOCK_FIELD_IS_BYREF:
        /*******
         // copy the onstack __block container to the heap
         __block ... x;
         __weak __block ... x;
         [^{ x; } copy];
         ********/
        
        _Block_byref_assign_copy(destAddr, object, flags);
        break;
        
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT:
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK:
        /*******
         // copy the actual field held in the __block container
         __block id object;
         __block void (^object)(void);
         [^{ object; } copy];
         ********/

        // under manual retain release __block object/block variables are dangling
        _Block_assign((void *)object, destAddr);
        break;

      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT | BLOCK_FIELD_IS_WEAK:
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK  | BLOCK_FIELD_IS_WEAK:
        /*******
         // copy the actual field held in the __block container
         __weak __block id object;
         __weak __block void (^object)(void);
         [^{ object; } copy];
         ********/

        _Block_assign_weak(object, destAddr);
        break;

      default:
        break;
    }
}

// When Blocks or Block_byrefs hold objects their destroy helper routines call this entry point
// to help dispose of the contents
// Used initially only for __attribute__((NSObject)) marked pointers.
BLOCK_EXPORT
void _Block_object_dispose(const void *object, const int flags) {
    switch (os_assumes(flags & BLOCK_ALL_COPY_DISPOSE_FLAGS)) {
      case BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_WEAK:
      case BLOCK_FIELD_IS_BYREF:
        // get rid of the __block data structure held in a Block
        _Block_byref_release(object);
        break;
      case BLOCK_FIELD_IS_BLOCK:
        _Block_destroy(object);
        break;
      case BLOCK_FIELD_IS_OBJECT:
        _Block_release_object(object);
        break;
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT:
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK:
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_OBJECT | BLOCK_FIELD_IS_WEAK:
      case BLOCK_BYREF_CALLER | BLOCK_FIELD_IS_BLOCK  | BLOCK_FIELD_IS_WEAK:
        break;
      default:
        break;
    }
}
