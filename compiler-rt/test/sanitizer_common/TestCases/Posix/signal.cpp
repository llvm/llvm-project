// RUN: %clangxx -std=c++11 -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <climits>
#include <errno.h>
#include <stdio.h>
#include <signal.h>

#include <initializer_list>

constexpr int std_signals[] = {
  SIGHUP,
  SIGINT,
  SIGQUIT,
  SIGILL,
  SIGTRAP,
  SIGABRT,
  SIGIOT,
  SIGBUS,
  SIGFPE,
  SIGUSR1,
  SIGSEGV,
  SIGUSR2,
  SIGPIPE,
  SIGALRM,
  SIGTERM,
  SIGCHLD,
  SIGCONT,
  SIGTSTP,
  SIGTTIN,
  SIGTTOU,
  SIGURG,
  SIGXCPU,
  SIGXFSZ,
  SIGVTALRM,
  SIGPROF,
  SIGWINCH,
  SIGIO,
  SIGSYS,
};

constexpr int no_change_act_signals[] = {
  SIGKILL,
  SIGSTOP,
};

void signal_handler(int) {}
void signal_action_handler(int, siginfo_t*, void*) {}

void test_signal_custom() {
  for (int signum : std_signals) {
    auto* ret = signal(signum, &signal_handler);
    assert(ret != SIG_ERR);
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    auto* ret = signal(signum, &signal_handler);
    assert(ret != SIG_ERR);
  }
#endif
  for (int signum : no_change_act_signals) {
    auto* ret = signal(signum, &signal_handler);
    int err = errno;
    assert(ret == SIG_ERR);
    assert(err == EINVAL);
  }
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    auto* ret = signal(signum, &signal_handler);
    int err = errno;
    assert(ret == SIG_ERR);
    assert(err == EINVAL);
  }
}

void test_signal_ignore() {
  for (int signum : std_signals) {
    auto* ret = signal(signum, SIG_IGN);
    if (signum != SIGCHLD) {
      // POSIX.1-1990 disallowed setting the action for SIGCHLD to SIG_IGN
      // though POSIX.1-2001 and later allow this possibility.
      assert(ret != SIG_ERR);
    }
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    auto* ret = signal(signum, SIG_IGN);
    assert(ret != SIG_ERR);
  }
#endif
  for (int signum : no_change_act_signals) {
    auto* ret = signal(signum, SIG_IGN);
    int err = errno;
    assert(ret == SIG_ERR);
    assert(err == EINVAL);
  }
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    auto* ret = signal(signum, SIG_IGN);
    int err = errno;
    assert(ret == SIG_ERR);
    assert(err == EINVAL);
  }
}

void test_signal_default() {
  for (int signum : std_signals) {
    auto* ret = signal(signum, SIG_DFL);
    assert(ret != SIG_ERR);
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    auto* ret = signal(signum, SIG_DFL);
    assert(ret != SIG_ERR);
  }
#endif
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    auto* ret = signal(signum, SIG_DFL);
    int err = errno;
    assert(ret == SIG_ERR);
    assert(err == EINVAL);
  }
}

void test_sigaction_custom() {
  struct sigaction act = {}, oldact;

  act.sa_handler = &signal_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;

  for (int signum : std_signals) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#endif
  for (int signum : no_change_act_signals) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }

  act.sa_handler = nullptr;
  act.sa_sigaction = &signal_action_handler;
  act.sa_flags = SA_SIGINFO;

  for (int signum : std_signals) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#endif
  for (int signum : no_change_act_signals) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
}

void test_sigaction_ignore() {
  struct sigaction act = {}, oldact;

  act.sa_handler = SIG_IGN;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;

  for (int signum : std_signals) {
    int ret = sigaction(signum, &act, &oldact);
    if (signum != SIGCHLD) {
      // POSIX.1-1990 disallowed setting the action for SIGCHLD to SIG_IGN
      // though POSIX.1-2001 and later allow this possibility.
      assert(ret == 0);
    }
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#endif
  for (int signum : no_change_act_signals) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
}

void test_sigaction_default() {
  struct sigaction act = {}, oldact;

  act.sa_handler = SIG_DFL;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;

  for (int signum : std_signals) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#ifdef SIGRTMIN
  for (int signum = SIGRTMIN; signum <= SIGRTMAX; ++signum) {
    int ret = sigaction(signum, &act, &oldact);
    assert(ret == 0);
  }
#endif
  for (int signum : {
        0,
#ifdef SIGRTMAX
        SIGRTMAX + 1,
#endif
        INT_MAX}) {
    int ret = sigaction(signum, &act, &oldact);
    int err = errno;
    assert(ret == -1);
    assert(err == EINVAL);
  }
}

int main(void) {
  printf("sigaction\n");

  test_signal_custom();
  test_signal_ignore();
  test_signal_default();

  test_sigaction_custom();
  test_sigaction_ignore();
  test_sigaction_default();

  // CHECK: sigaction

  return 0;
}
