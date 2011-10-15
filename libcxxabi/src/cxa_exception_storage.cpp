//===--------------------- cxa_exception_storage.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//  
//  This file implements the storage for the "Caught Exception Stack"
//  http://www.codesourcery.com/public/cxx-abi/abi-eh.html (section 2.2.2)
//  
//===----------------------------------------------------------------------===//

#include "cxa_exception.hpp"

#ifdef HAS_THREAD_LOCAL

namespace __cxxabiv1 {

namespace {
    __cxa_eh_globals * __globals () throw () {
        static thread_local __cxa_eh_globals eh_globals;
        return &eh_globals;
        }
    }

extern "C" {
    __cxa_eh_globals * __cxa_get_globals      () throw() { return __globals (); }
    __cxa_eh_globals * __cxa_get_globals_fast () throw() { return __globals (); }
    }
}

#else

#include <pthread.h>
#include <cstdlib>          // for calloc, free
#include "abort_message.h"

//  In general, we treat all pthread errors as fatal.
//  We cannot call std::terminate() because that will in turn
//  call __cxa_get_globals() and cause infinite recursion.

namespace __cxxabiv1 {
namespace {
    pthread_once_t flag_ = PTHREAD_ONCE_INIT;
    pthread_key_t  key_;

    void destruct_ (void *p) throw () {
        std::free ( p );
        if ( 0 != ::pthread_setspecific ( key_, NULL ) ) 
            abort_message("cannot zero out thread value for __cxa_get_globals()");
        }

    void construct_ () throw () {
        if ( 0 != pthread_key_create ( &key_, destruct_ ) )
            abort_message("cannot create pthread key for __cxa_get_globals()");
        }
}   

extern "C" {
    __cxa_eh_globals * __cxa_get_globals () throw () {
    //  First time through, create the key.
        if ( 0 != pthread_once ( &flag_, construct_ ) ) 
            abort_message("cannot run pthread_once for __cxa_get_globals()");

    //  Try to get the globals for this thread
        __cxa_eh_globals* retVal = __cxa_get_globals_fast ();
    
    //  If this is the first time we've been asked for these globals, create them
        if ( NULL == retVal ) {
            retVal = static_cast<__cxa_eh_globals*>
                        (std::calloc (1, sizeof (__cxa_eh_globals)));
            if ( NULL == retVal )
                abort_message("cannot allocate __cxa_eh_globals");
            if ( 0 != pthread_setspecific ( key_, retVal ) )
               abort_message("pthread_setspecific failure in __cxa_get_globals()");
           }
        return retVal;
        }

    __cxa_eh_globals * __cxa_get_globals_fast () throw () {
        return static_cast<__cxa_eh_globals*>(::pthread_getspecific(key_));
        }
    
}
}
#endif
