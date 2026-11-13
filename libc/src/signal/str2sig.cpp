//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of str2sig.
///
//===----------------------------------------------------------------------===//

#include "src/signal/str2sig.h"
#include "hdr/signal_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/str_to_integer.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

struct SigNameMapping {
  int signum;
  const char *name;
};

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

LLVM_LIBC_FUNCTION(int, str2sig,
                   (const char *__restrict str, int *__restrict signump)) {
  if (str == nullptr || signump == nullptr || *str == '\0')
    return -1;

  cpp::string_view name(str);

  for (const auto &mapping : SIGNAL_NAMES) {
    if (name == mapping.name) {
      *signump = mapping.signum;
      return 0;
    }
  }

#if defined(SIGRTMIN) && defined(SIGRTMAX)
  if (name == "RTMIN") {
    *signump = SIGRTMIN;
    return 0;
  }
  if (name == "RTMAX") {
    *signump = SIGRTMAX;
    return 0;
  }
  if (name.starts_with("RTMIN+")) {
    auto parse_res = internal::strtointeger<int>(str + 6, 10);
    if (parse_res.has_error())
      return -1;
    int offset = parse_res.value();
    if (offset < 0 || SIGRTMIN + offset > SIGRTMAX)
      return -1;
    *signump = SIGRTMIN + offset;
    return 0;
  }
  if (name.starts_with("RTMAX-")) {
    auto parse_res = internal::strtointeger<int>(str + 6, 10);
    if (parse_res.has_error())
      return -1;
    int offset = parse_res.value();
    if (offset < 0 || SIGRTMAX - offset < SIGRTMIN)
      return -1;
    *signump = SIGRTMAX - offset;
    return 0;
  }
#endif

  // POSIX also allows decimal integer string representation of signals.
  auto parse_res = internal::strtointeger<int>(str, 10);
  if (!parse_res.has_error()) {
    int num = parse_res.value();
    if (num > 0 && num <= NSIG) {
      *signump = num;
      return 0;
    }
  }

  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
