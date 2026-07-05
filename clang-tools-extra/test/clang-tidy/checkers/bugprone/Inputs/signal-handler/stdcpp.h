//===--- Header for test bugprone-signal-handler.cpp ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define SIGINT 1
#define SIG_IGN std::_sig_ign
#define SIG_DFL std::_sig_dfl

namespace std {

void _sig_ign(int);
void _sig_dfl(int);

typedef void (*sighandler_t)(int);
sighandler_t signal(int, sighandler_t);

void abort();
void _Exit(int);
void quick_exit(int);

void other_call();

struct SysStruct {
  void operator<<(int);
};

} // namespace std

namespace system_other {

typedef void (*sighandler_t)(int);
sighandler_t signal(int, sighandler_t);

} // namespace system_other
