//===------------------------ cxa_aux_runtime.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//
// This file implements the "Auxiliary Runtime APIs"
// http://mentorembedded.github.io/cxx-abi/abi-eh.html#cxx-aux
//===----------------------------------------------------------------------===//

#include "cxxabi.h"
#include <typeinfo>

namespace __cxxabiv1
{

extern "C"
{

LIBCXXABI_NORETURN
void __cxa_bad_cast (void) {
    throw std::bad_cast();
}

LIBCXXABI_NORETURN
void __cxa_bad_typeid(void) {
    throw std::bad_typeid();
}

}  // extern "C"

}  // abi
