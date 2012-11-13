//===------------------------- cxa_handlers.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
// This file implements the functionality associated with the terminate_handler,
//   unexpected_handler, and new_handler.
//===----------------------------------------------------------------------===//

#include <stdexcept>
#include <new>
#include <exception>
#include "abort_message.h"
#include "cxxabi.h"
#include "cxa_handlers.hpp"
#include "cxa_exception.hpp"
#include "private_typeinfo.h"

namespace std
{

unexpected_handler
get_unexpected() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_unexpected_handler, (unexpected_handler)0);
//  The above is safe but overkill on x86
//  Using of C++11 atomics this should be rewritten
//  return __cxa_unexpected_handler.load(memory_order_acq);
}

__attribute__((visibility("hidden"), noreturn))
void
__unexpected(unexpected_handler func)
{
    func();
    // unexpected handler should not return
    abort_message("unexpected_handler unexpectedly returned");
}

__attribute__((noreturn))
void
unexpected()
{
    __unexpected(get_unexpected());
}

terminate_handler
get_terminate() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_terminate_handler, (terminate_handler)0);
//  The above is safe but overkill on x86
//  Using of C++11 atomics this should be rewritten
//  return __cxa_terminate_handler.load(memory_order_acq);
}

__attribute__((visibility("hidden"), noreturn))
void
__terminate(terminate_handler func) _NOEXCEPT
{
#if __has_feature(cxx_exceptions)
    try
    {
#endif  // __has_feature(cxx_exceptions)
        func();
        // handler should not return
        abort_message("terminate_handler unexpectedly returned");
#if __has_feature(cxx_exceptions)
    }
    catch (...)
    {
        // handler should not throw exception
        abort_message("terminate_handler unexpectedly threw an exception");
    }
#endif  // #if __has_feature(cxx_exceptions)
}

__attribute__((noreturn))
void
terminate() _NOEXCEPT
{
    // If there might be an uncaught exception
    using namespace __cxxabiv1;
    __cxa_eh_globals* globals = __cxa_get_globals_fast();
    if (globals)
    {
        __cxa_exception* exception_header = globals->caughtExceptions;
        if (exception_header)
        {
            _Unwind_Exception* unwind_exception =
                reinterpret_cast<_Unwind_Exception*>(exception_header + 1) - 1;
            bool native_exception =
                (unwind_exception->exception_class & get_vendor_and_language) ==
                               (kOurExceptionClass & get_vendor_and_language);
            if (native_exception)
                __terminate(exception_header->terminateHandler);
        }
    }
    __terminate(get_terminate());
}

extern "C" new_handler __cxa_new_handler = 0;
// In the future these will become:
// std::atomic<std::new_handler>  __cxa_new_handler(0);

new_handler
set_new_handler(new_handler handler) _NOEXCEPT
{
    return __sync_swap(&__cxa_new_handler, handler);
//  Using of C++11 atomics this should be rewritten
//  return __cxa_new_handler.exchange(handler, memory_order_acq_rel);
}

new_handler
get_new_handler() _NOEXCEPT
{
    return __sync_fetch_and_add(&__cxa_new_handler, (new_handler)0);
//  The above is safe but overkill on x86
//  Using of C++11 atomics this should be rewritten
//  return __cxa_new_handler.load(memory_order_acq);
}

}  // std
