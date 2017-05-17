//===------------------------- cxa_exception.hpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//  
//  This file implements the "Exception Handling APIs"
//  http://mentorembedded.github.io/cxx-abi/abi-eh.html
//  
//===----------------------------------------------------------------------===//

#ifndef _CXA_EXCEPTION_H
#define _CXA_EXCEPTION_H

#include <exception> // for std::unexpected_handler and std::terminate_handler
#include "cxxabi.h"
#include "unwind.h"

namespace __cxxabiv1 {

static const uint64_t kOurExceptionClass          = 0x434C4E47432B2B00; // CLNGC++\0
static const uint64_t kOurDependentExceptionClass = 0x434C4E47432B2B01; // CLNGC++\1
static const uint64_t get_vendor_and_language     = 0xFFFFFFFFFFFFFF00; // mask for CLNGC++

struct _LIBCXXABI_HIDDEN __cxa_exception {
#if defined(__LP64__) || defined(_LIBCXXABI_ARM_EHABI)
    // This is a new field to support C++ 0x exception_ptr.
    // For binary compatibility it is at the start of this
    // struct which is prepended to the object thrown in
    // __cxa_allocate_exception.
    size_t referenceCount;
#endif

    //  Manage the exception object itself.
    std::type_info *exceptionType;
    void (*exceptionDestructor)(void *);
    std::unexpected_handler unexpectedHandler;
    std::terminate_handler  terminateHandler;

    __cxa_exception *nextException;

    int handlerCount;

#if defined(_LIBCXXABI_ARM_EHABI)
    __cxa_exception* nextPropagatingException;
    int propagationCount;
#else
    int handlerSwitchValue;
    const unsigned char *actionRecord;
    const unsigned char *languageSpecificData;
    void *catchTemp;
    void *adjustedPtr;
#endif

#if !defined(__LP64__) && !defined(_LIBCXXABI_ARM_EHABI)
    // This is a new field to support C++ 0x exception_ptr.
    // For binary compatibility it is placed where the compiler
    // previously adding padded to 64-bit align unwindHeader.
    size_t referenceCount;
#endif

    // This field is annotated with attribute aligned so that the exception
    // object following the field is sufficiently aligned and there is no
    // gap between the field and the exception object. r276215 made a change to
    // annotate _Unwind_Exception in unwind.h with __attribute__((aligned)), but
    // we cannot incorporate the fix on Darwin since it is an ABI-breaking
    // change, which is why we need the attribute on this field.
    //
    // For ARM EHABI, we do not align this field since _Unwind_Exception is an
    // alias of _Unwind_Control_Block, which is not annotated with
    // __attribute__((aligned).
#if defined(_LIBCXXABI_ARM_EHABI)
    _Unwind_Exception unwindHeader;
#else
    _Unwind_Exception unwindHeader __attribute__((aligned));
#endif
};

// http://sourcery.mentor.com/archives/cxx-abi-dev/msg01924.html
// The layout of this structure MUST match the layout of __cxa_exception, with
// primaryException instead of referenceCount.
struct _LIBCXXABI_HIDDEN __cxa_dependent_exception {
#if defined(__LP64__) || defined(_LIBCXXABI_ARM_EHABI)
    void* primaryException;
#endif

    std::type_info *exceptionType;
    void (*exceptionDestructor)(void *);
    std::unexpected_handler unexpectedHandler;
    std::terminate_handler terminateHandler;

    __cxa_exception *nextException;

    int handlerCount;

#if defined(_LIBCXXABI_ARM_EHABI)
    __cxa_exception* nextPropagatingException;
    int propagationCount;
#else
    int handlerSwitchValue;
    const unsigned char *actionRecord;
    const unsigned char *languageSpecificData;
    void * catchTemp;
    void *adjustedPtr;
#endif

#if !defined(__LP64__) && !defined(_LIBCXXABI_ARM_EHABI)
    void* primaryException;
#endif

    // See the comment in __cxa_exception as to why this field has attribute
    // aligned.
#if defined(_LIBCXXABI_ARM_EHABI)
    _Unwind_Exception unwindHeader;
#else
    _Unwind_Exception unwindHeader __attribute__((aligned));
#endif
};

struct _LIBCXXABI_HIDDEN __cxa_eh_globals {
    __cxa_exception *   caughtExceptions;
    unsigned int        uncaughtExceptions;
#if defined(_LIBCXXABI_ARM_EHABI)
    __cxa_exception* propagatingExceptions;
#endif
};

extern "C" _LIBCXXABI_FUNC_VIS __cxa_eh_globals * __cxa_get_globals      ();
extern "C" _LIBCXXABI_FUNC_VIS __cxa_eh_globals * __cxa_get_globals_fast ();

extern "C" _LIBCXXABI_FUNC_VIS void * __cxa_allocate_dependent_exception ();
extern "C" _LIBCXXABI_FUNC_VIS void __cxa_free_dependent_exception (void * dependent_exception);

}  // namespace __cxxabiv1

#endif  // _CXA_EXCEPTION_H
