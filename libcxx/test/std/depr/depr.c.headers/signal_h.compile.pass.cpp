//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <signal.h>

#include <signal.h>

#include "test_macros.h"

#ifndef SIG_DFL
#error SIG_DFL not defined
#endif

#ifndef SIG_ERR
#error SIG_ERR not defined
#endif

#ifndef SIG_IGN
#error SIG_IGN not defined
#endif

#ifndef SIGABRT
#error SIGABRT not defined
#endif

#ifndef SIGFPE
#error SIGFPE not defined
#endif

#ifndef SIGILL
#error SIGILL not defined
#endif

#ifndef SIGINT
#error SIGINT not defined
#endif

#ifndef SIGSEGV
#error SIGSEGV not defined
#endif

#ifndef SIGTERM
#error SIGTERM not defined
#endif

sig_atomic_t sig;
typedef void (*func)(int);
ASSERT_SAME_TYPE(func, decltype(signal(0, (func)0)));
ASSERT_SAME_TYPE(int,  decltype(raise(0)));
