// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//


#ifndef _Block_H_
#define _Block_H_

#if defined(_WIN32)
#    if defined(BlocksRuntime_STATIC)
#        define BLOCK_ABI
#    else
#        if defined(BlocksRuntime_EXPORTS)
#            define BLOCK_ABI __declspec(dllexport)
#        else
#            define BLOCK_ABI __declspec(dllimport)
#        endif
#    endif
#else
#    define BLOCK_ABI __attribute__((__visibility__("default")))
#endif

#if !defined(BLOCK_EXPORT)
#   if defined(__cplusplus)
#       define BLOCK_EXPORT extern "C" BLOCK_ABI
#   else
#       define BLOCK_EXPORT extern BLOCK_ABI
#   endif
#endif

#if __cplusplus
extern "C" {
#endif

// Create a heap based copy of a Block or simply add a reference to an existing one.
// This must be paired with Block_release to recover memory, even when running
// under Objective-C Garbage Collection.
BLOCK_EXPORT void *_Block_copy(const void *aBlock);

// Lose the reference, and if heap based and last reference, recover the memory
BLOCK_EXPORT void _Block_release(const void *aBlock);

// Used by the compiler. Do not call this function yourself.
BLOCK_EXPORT void _Block_object_assign(void *, const void *, const int);

// Used by the compiler. Do not call this function yourself.
BLOCK_EXPORT void _Block_object_dispose(const void *, const int);

// Used by the compiler. Do not use these variables yourself.
#if defined(_WIN32)
extern void * _NSConcreteGlobalBlock[32];
extern void * _NSConcreteStackBlock[32];
#else
BLOCK_EXPORT void * _NSConcreteGlobalBlock[32];
BLOCK_EXPORT void * _NSConcreteStackBlock[32];
#endif

#if __cplusplus
}
#endif

// Type correct macros

#define Block_copy(...) ((__typeof(__VA_ARGS__))_Block_copy((const void *)(__VA_ARGS__)))
#define Block_release(...) _Block_release((const void *)(__VA_ARGS__))


#endif
