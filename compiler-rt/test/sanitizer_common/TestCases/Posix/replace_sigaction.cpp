// RUN: %clangxx -std=c++11 -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <signal.h>
#include <unistd.h>

static bool sahandler_done;
static bool sasigaction_done;

static void sahandler(int) { sahandler_done = true; }

static void sasigaction(int, siginfo_t *, void *) { sasigaction_done = true; }

template <typename T> void install(T *handler, struct sigaction *prev) {
  bool siginfo = handler == (T *)&sasigaction;
  struct sigaction act = {};
  if (siginfo) {
    act.sa_flags = SA_SIGINFO;
    act.sa_sigaction = (decltype(act.sa_sigaction))handler;
  } else {
    act.sa_handler = (decltype(act.sa_handler))handler;
  }
  int ret = sigaction(SIGHUP, &act, prev);
  assert(ret == 0);

  if (handler == (T *)&sahandler) {
    sahandler_done = false;
    raise(SIGHUP);
    assert(sahandler_done);
  }

  if (handler == (T *)&sasigaction) {
    sasigaction_done = false;
    raise(SIGHUP);
    assert(sasigaction_done);
  }
}

template <typename T1, typename T2> void test(T1 *from, T2 *to) {
  install(from, nullptr);
  struct sigaction prev = {};
  install(to, &prev);

  bool siginfo_from = (from == (T1 *)&sasigaction);
  if (siginfo_from) {
    assert(prev.sa_flags & SA_SIGINFO);
    assert(prev.sa_sigaction == (decltype(prev.sa_sigaction))from);
  } else {
    assert((prev.sa_flags & SA_SIGINFO) == 0);
    assert(prev.sa_handler == (decltype(prev.sa_handler))from);
  }
}

template <typename T> void testAll(T *to) {
  test(&sahandler, to);
  test(&sasigaction, to);
  test(SIG_IGN, to);
  test(SIG_DFL, to);
}

int main(void) {
  testAll(&sahandler);
  testAll(&sasigaction);
  testAll(SIG_IGN);
  testAll(SIG_DFL);
}
