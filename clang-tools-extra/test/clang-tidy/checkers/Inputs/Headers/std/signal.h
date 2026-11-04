//===--- signal.h - Stub header for tests -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SIGNAL_H_
#define _SIGNAL_H_

typedef void (*sighandler_t)(int);

void _sig_ign(int);
void _sig_dfl(int);
void _sig_err(int);

#define SIGINT 1
#define SIG_IGN _sig_ign
#define SIG_DFL _sig_dfl
#define SIG_ERR _sig_err

sighandler_t signal(int, sighandler_t);

#endif // _SIGNAL_H_
