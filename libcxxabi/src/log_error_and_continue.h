//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LOG_ERROR_AND_CONTINUE_H_
#define __LOG_ERROR_AND_CONTINUE_H_

#include "cxxabi.h"

extern "C" _LIBCXXABI_HIDDEN void __log_error_and_continue(const char* message);

#endif // __LOG_ERROR_AND_CONTINUE_H_
