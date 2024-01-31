//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <deque>
#include <cassert>
#include <inttypes.h>

// TODO: Temporary workaround, see https://github.com/llvm/llvm-project/pull/79654#issuecomment-1919397302
#if __has_include(<__thread/support.h>)
#  include <__thread/support.h>
#else
#  include <__threading_support>
#endif

// UNSUPPORTED: c++03
// UNSUPPORTED: modules-build && no-threads

// Necessary because we include a private source file of libc++abi, which
// only understands _LIBCXXABI_HAS_NO_THREADS.
#include "test_macros.h"
#ifdef TEST_HAS_NO_THREADS
# define _LIBCXXABI_HAS_NO_THREADS
#endif

typedef std::deque<void *> container;

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#define _LIBCXXABI_ASSERT(expr, msg) assert((expr) && (msg))

// #define  DEBUG_FALLBACK_MALLOC
#define INSTRUMENT_FALLBACK_MALLOC
#include "../src/fallback_malloc.cpp"
TEST_DIAGNOSTIC_POP

void assertAlignment(void* ptr) { assert(reinterpret_cast<size_t>(ptr) % alignof(FallbackMaxAlignType) == 0); }

container alloc_series ( size_t sz ) {
    container ptrs;
    void *p;

    while (NULL != (p = fallback_malloc(sz))) {
      assertAlignment(p);
      ptrs.push_back(p);
    }
    return ptrs;
}

container alloc_series ( size_t sz, float growth ) {
    container ptrs;
    void *p;

    while ( NULL != ( p = fallback_malloc ( sz ))) {
      assertAlignment(p);
      ptrs.push_back(p);
      sz *= growth;
    }

    return ptrs;
}

container alloc_series ( const size_t *first, size_t len ) {
    container ptrs;
    const size_t *last = first + len;
    void * p;

    for ( const size_t *iter = first; iter != last; ++iter ) {
        if ( NULL == (p = fallback_malloc ( *iter )))
            break;
        assertAlignment(p);
        ptrs.push_back ( p );
    }

    return ptrs;
}

void *pop ( container &c, bool from_end ) {
    void *ptr;
    if ( from_end ) {
        ptr = c.back ();
        c.pop_back ();
    }
    else {
        ptr = c.front ();
        c.pop_front ();
    }
    return ptr;
}

void exhaustion_test1 () {
    container ptrs;

    init_heap ();
    std::printf("Constant exhaustion tests\n");

//  Delete in allocation order
    ptrs = alloc_series ( 32 );
    std::printf("Allocated %zu 32 byte chunks\n", ptrs.size());
    print_free_list ();
    for ( container::iterator iter = ptrs.begin (); iter != ptrs.end (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Delete in reverse order
    ptrs = alloc_series ( 32 );
    std::printf("Allocated %zu 32 byte chunks\n", ptrs.size());
    for ( container::reverse_iterator iter = ptrs.rbegin (); iter != ptrs.rend (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Alternate deletions
    ptrs = alloc_series ( 32 );
    std::printf("Allocated %zu 32 byte chunks\n", ptrs.size());
    while ( ptrs.size () > 0 )
        fallback_free ( pop ( ptrs, ptrs.size () % 1 == 1 ));
    print_free_list ();
}

void exhaustion_test2 () {
    container ptrs;
    init_heap ();

    std::printf("Growing exhaustion tests\n");

//  Delete in allocation order
    ptrs = alloc_series ( 32, 1.5 );

    std::printf("Allocated %zu { 32, 48, 72, 108, 162 ... } byte chunks\n",
                ptrs.size());
    print_free_list ();
    for ( container::iterator iter = ptrs.begin (); iter != ptrs.end (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Delete in reverse order
    print_free_list ();
    ptrs = alloc_series ( 32, 1.5 );
    std::printf("Allocated %zu { 32, 48, 72, 108, 162 ... } byte chunks\n",
                ptrs.size());
    for ( container::reverse_iterator iter = ptrs.rbegin (); iter != ptrs.rend (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Alternate deletions
    ptrs = alloc_series ( 32, 1.5 );
    std::printf("Allocated %zu { 32, 48, 72, 108, 162 ... } byte chunks\n",
                ptrs.size());
    while ( ptrs.size () > 0 )
        fallback_free ( pop ( ptrs, ptrs.size () % 1 == 1 ));
    print_free_list ();

}

void exhaustion_test3 () {
    const size_t allocs [] = { 124, 60, 252, 60, 4 };
    container ptrs;
    init_heap ();

    std::printf("Complete exhaustion tests\n");

//  Delete in allocation order
    ptrs = alloc_series ( allocs, sizeof ( allocs ) / sizeof ( allocs[0] ));
    std::printf("Allocated %zu chunks\n", ptrs.size());
    print_free_list ();
    for ( container::iterator iter = ptrs.begin (); iter != ptrs.end (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Delete in reverse order
    print_free_list ();
    ptrs = alloc_series ( allocs, sizeof ( allocs ) / sizeof ( allocs[0] ));
    std::printf("Allocated %zu chunks\n", ptrs.size());
    for ( container::reverse_iterator iter = ptrs.rbegin (); iter != ptrs.rend (); ++iter )
        fallback_free ( *iter );
    print_free_list ();
    std::printf("----\n");

//  Alternate deletions
    ptrs = alloc_series ( allocs, sizeof ( allocs ) / sizeof ( allocs[0] ));
    std::printf("Allocated %zu chunks\n", ptrs.size());
    while ( ptrs.size () > 0 )
        fallback_free ( pop ( ptrs, ptrs.size () % 1 == 1 ));
    print_free_list ();

}


int main () {
    print_free_list ();

    char *p = (char *) fallback_malloc ( 1024 );    // too big!
    std::printf("fallback_malloc ( 1024 ) --> %" PRIuPTR"\n", (uintptr_t) p);
    print_free_list ();

    p = (char *) fallback_malloc ( 32 );
    std::printf("fallback_malloc ( 32 ) --> %" PRIuPTR"\n", (uintptr_t) (p - heap));
    if ( !is_fallback_ptr ( p ))
        std::printf("### p is not a fallback pointer!!\n");

    print_free_list ();
    fallback_free ( p );
    print_free_list ();

    exhaustion_test1();
    exhaustion_test2();
    exhaustion_test3();
    return 0;
}
