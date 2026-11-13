//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of sig2str.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sig2str.h"
#include "hdr/signal_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/StringUtil/signal_to_string.h"
#include "src/__support/common.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

struct SigNameMapping {
  int signum;
  const char *name;
};

// POSIX standard signal names without "SIG" prefix.
constexpr SigNameMapping SIGNAL_NAMES[] = {
#ifdef SIGHUP
    {SIGHUP, "HUP"},
#endif
#ifdef SIGINT
    {SIGINT, "INT"},
#endif
#ifdef SIGQUIT
    {SIGQUIT, "QUIT"},
#endif
#ifdef SIGILL
    {SIGILL, "ILL"},
#endif
#ifdef SIGTRAP
    {SIGTRAP, "TRAP"},
#endif
#ifdef SIGABRT
    {SIGABRT, "ABRT"},
#endif
#ifdef SIGBUS
    {SIGBUS, "BUS"},
#endif
#ifdef SIGFPE
    {SIGFPE, "FPE"},
#endif
#ifdef SIGKILL
    {SIGKILL, "KILL"},
#endif
#ifdef SIGUSR1
    {SIGUSR1, "USR1"},
#endif
#ifdef SIGSEGV
    {SIGSEGV, "SEGV"},
#endif
#ifdef SIGUSR2
    {SIGUSR2, "USR2"},
#endif
#ifdef SIGPIPE
    {SIGPIPE, "PIPE"},
#endif
#ifdef SIGALRM
    {SIGALRM, "ALRM"},
#endif
#ifdef SIGTERM
    {SIGTERM, "TERM"},
#endif
#ifdef SIGSTKFLT
    {SIGSTKFLT, "STKFLT"},
#endif
#ifdef SIGCHLD
    {SIGCHLD, "CHLD"},
#endif
#ifdef SIGCONT
    {SIGCONT, "CONT"},
#endif
#ifdef SIGSTOP
    {SIGSTOP, "STOP"},
#endif
#ifdef SIGTSTP
    {SIGTSTP, "TSTP"},
#endif
#ifdef SIGTTIN
    {SIGTTIN, "TTIN"},
#endif
#ifdef SIGTTOU
    {SIGTTOU, "TTOU"},
#endif
#ifdef SIGURG
    {SIGURG, "URG"},
#endif
#ifdef SIGXCPU
    {SIGXCPU, "XCPU"},
#endif
#ifdef SIGXFSZ
    {SIGXFSZ, "XFSZ"},
#endif
#ifdef SIGVTALRM
    {SIGVTALRM, "VTALRM"},
#endif
#ifdef SIGPROF
    {SIGPROF, "PROF"},
#endif
#ifdef SIGWINCH
    {SIGWINCH, "WINCH"},
#endif
#ifdef SIGPOLL
    {SIGPOLL, "POLL"},
#endif
#ifdef SIGPWR
    {SIGPWR, "PWR"},
#endif
#ifdef SIGSYS
    {SIGSYS, "SYS"},
#endif
};

} // namespace

LLVM_LIBC_FUNCTION(int, sig2str, (int signum, char *str)) {
  if (str == nullptr)
    return -1;

  for (const auto &mapping : SIGNAL_NAMES) {
    if (mapping.signum == signum) {
      char *dest = str;
      for (const char *s = mapping.name; *s != '\0'; ++s, ++dest)
        *dest = *s;
      *dest = '\0';
      return 0;
    }
  }

#if defined(SIGRTMIN) && defined(SIGRTMAX)
  if (signum == SIGRTMIN) {
    str[0] = 'R';
    str[1] = 'T';
    str[2] = 'M';
    str[3] = 'I';
    str[4] = 'N';
    str[5] = '\0';
    return 0;
  }
  if (signum == SIGRTMAX) {
    str[0] = 'R';
    str[1] = 'T';
    str[2] = 'M';
    str[3] = 'A';
    str[4] = 'X';
    str[5] = '\0';
    return 0;
  }
  int rt_mid = SIGRTMIN + (SIGRTMAX - SIGRTMIN) / 2;
  if (signum > SIGRTMIN && signum <= rt_mid) {
    str[0] = 'R';
    str[1] = 'T';
    str[2] = 'M';
    str[3] = 'I';
    str[4] = 'N';
    str[5] = '+';
    char *dest = str + 6;
    int offset = signum - SIGRTMIN;
    IntegerToString<int> str_num(offset);
    for (char c : str_num.view())
      *dest++ = c;
    *dest = '\0';
    return 0;
  }
  if (signum > rt_mid && signum < SIGRTMAX) {
    str[0] = 'R';
    str[1] = 'T';
    str[2] = 'M';
    str[3] = 'A';
    str[4] = 'X';
    str[5] = '-';
    char *dest = str + 6;
    int offset = SIGRTMAX - signum;
    IntegerToString<int> str_num(offset);
    for (char c : str_num.view())
      *dest++ = c;
    *dest = '\0';
    return 0;
  }
#endif

  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
