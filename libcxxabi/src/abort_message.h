//===-------------------------- abort_message.h-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __ABORT_MESSAGE_H_
#define __ABORT_MESSAGE_H_

#include "cxxabi.h"

#ifdef	__cplusplus
extern "C" {
#endif

extern LIBCXXABI_NORETURN
       void abort_message(const char* format, ...) 
            __attribute__((format(printf, 1, 2)));


#ifdef __cplusplus
}
#endif


#endif

