// REQUIRES: !asan, compiler-rt, lldb
// UNSUPPORTED: system-windows
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
// UNSUPPORTED: apple-lldb-pre-1000

// XFAIL: lldb
// lldb-8, even outside of dexter, will sometimes trigger an asan fault in
// the debugged process and generally freak out.

// RUN: %clang++ -std=gnu++11 -O1 -glldb -fsanitize=address -arch x86_64 %s \
// RUN:   -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
#include <deque>

struct A {
  int a;
  A(int a) : a(a) {}
  A() : a(0) {}
};

using deq_t = std::deque<A>;

template class std::deque<A>;

static void __attribute__((noinline, optnone)) escape(deq_t &deq) {
  static volatile deq_t *sink;
  sink = &deq;
}

int main() {
  deq_t deq;
  deq.push_back(1234);
  deq.push_back(56789);
  escape(deq); // !dex_label first
  while (!deq.empty()) {
    auto record = deq.front();
    deq.pop_front();
    escape(deq); // !dex_label second
  }
}

// CHECK-DAG: seen_values: 3
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label first}:
  !value deq:
    "[0]":
      a: 1234
    "[1]":
      a: 56789
!where {lines: !label second}:
  !value deq:
    "[0]":
      a: 56789
...
*/
