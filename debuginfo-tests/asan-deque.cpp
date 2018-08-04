// RUN: %clangxx -arch x86_64 %target_itanium_abi_host_triple -O1 -g %s -o %t.out -fsanitize=address
// RUN: %test_debuginfo %s %t.out
// REQUIRES: not_asan
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
// UNSUPPORTED: apple-lldb-pre-1000
#include <deque>

struct A {
  int a;
  A(int a) : a(a) {}
};

using log_t = std::deque<A>;

static void __attribute__((noinline, optnone)) escape(log_t &log) {
  static volatile log_t *sink;
  sink = &log;
}

int main() {
  log_t log;
  log.push_back(1234);
  log.push_back(56789);
  escape(log);
  // DEBUGGER: break 25
  while (!log.empty()) {
    auto record = log.front();
    log.pop_front();
    escape(log);
    // DEBUGGER: break 30
  }
}

// DEBUGGER: r

// (at line 25)
// DEBUGGER: p log
// CHECK: 1234
// CHECK: 56789

// DEBUGGER: c

// (at line 30)
// DEBUGGER: p log
// CHECK: 56789
