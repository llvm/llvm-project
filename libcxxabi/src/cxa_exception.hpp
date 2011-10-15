#include <exception> // for std::unexpected_handler and std::terminate_handler
#include <cxxabi.h>
#include "unwind.h"

namespace __cxxabiv1 {

    struct __cxa_exception { 
#if __LP64__
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
    
#ifdef __ARM_EABI_UNWINDER__
         __cxa_exception* nextPropagatingException;
         int propagationCount;
#else
         int handlerSwitchValue;
         const unsigned char *actionRecord;
         const unsigned char *languageSpecificData;
         void *catchTemp;
         void *adjustedPtr;
#endif

#if !__LP64__
    // This is a new field to support C++ 0x exception_ptr.
    // For binary compatibility it is placed where the compiler
    // previously adding padded to 64-bit align unwindHeader.
        size_t referenceCount;
#endif
    
        _Unwind_Exception unwindHeader;
        };
    
    struct __cxa_dependent_exception {
#if __LP64__
        void* primaryException;
#endif
    
    // Unused dummy data (should be set to null)
        std::type_info *exceptionType;
        void (*exceptionDestructor)(void *); 
    
        std::unexpected_handler unexpectedHandler;
        std::terminate_handler terminateHandler;
    
        __cxa_exception *nextException;
    
        int handlerCount;
    
#ifdef __ARM_EABI_UNWINDER__
        __cxa_exception* nextPropagatingException;
        int propagationCount;
#else
        int handlerSwitchValue;
        const unsigned char *actionRecord;
        const unsigned char *languageSpecificData;
        void * catchTemp;
        void *adjustedPtr;
#endif
    
#if !__LP64__
        void* primaryException;
#endif
    
        _Unwind_Exception unwindHeader;
        };
        
    struct __cxa_eh_globals {
        __cxa_exception *   caughtExceptions;
        unsigned int        uncaughtExceptions;
#ifdef __ARM_EABI_UNWINDER__
        __cxa_exception* propagatingExceptions;
#endif
    };

    extern "C" __cxa_eh_globals * __cxa_get_globals      () throw();
    extern "C" __cxa_eh_globals * __cxa_get_globals_fast () throw();

    extern "C" void * __cxa_allocate_dependent_exception () throw();
    extern "C" void __cxa_free_dependent_exception (void * dependent_exception) throw();

}