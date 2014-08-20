//===-------------------------- abort_message.h-----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __ABORT_MESSAGE_H_
#define __ABORT_MESSAGE_H_

#include <stdio.h>

#pragma GCC visibility push(hidden)

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("hidden"), noreturn))
       void abort_message(const char* format, ...) 
            __attribute__((format(printf, 1, 2)));


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif

