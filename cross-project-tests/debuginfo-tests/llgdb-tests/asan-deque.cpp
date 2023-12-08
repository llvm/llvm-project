// RUN: %clangxx -arch x86_64 %target_itanium_abi_host_triple -O1 -g %s -o %t.out -fsanitize=address
// RUN: %test_debuginfo %s %t.out
// REQUIRES: !asan, compiler-rt, system-darwin
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
//           gdb is used on non-darwin; some configs pretty print std::deque,
//           some don't.
// UNSUPPORTED: apple-lldb-pre-1000
// XFAIL: !system-darwin && gdb-clang-incompatibility
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
  // DEBUGGER: break 28
  while (!log.empty()) {
    auto record = log.front();
    log.pop_front();
    escape(log);
    // DEBUGGER: break 33
  }
}

// DEBUGGER: r

// (at line 28)
// DEBUGGER: p log
// CHECK: 1234
// CHECK: 56789

// DEBUGGER: c

// (at line 33)
// DEBUGGER: p log
// CHECK: 56789
