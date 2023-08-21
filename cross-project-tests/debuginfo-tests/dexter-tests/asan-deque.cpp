// REQUIRES: !asan, compiler-rt, lldb
// UNSUPPORTED: system-windows
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
// UNSUPPORTED: apple-lldb-pre-1000

// XFAIL: lldb
// lldb-8, even outside of dexter, will sometimes trigger an asan fault in
// the debugged process and generally freak out.

// RUN: %clang -O1 -glldb -fsanitize=address -arch x86_64 %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary %t --debugger 'lldb' -- %s
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
  escape(deq); // DexLabel('first')
  while (!deq.empty()) {
    auto record = deq.front();
    deq.pop_front();
    escape(deq); // DexLabel('second')
  }
}

// DexExpectWatchValue('deq[0].a', '1234', on_line=ref('first'))
// DexExpectWatchValue('deq[1].a', '56789', on_line=ref('first'))

// DexExpectWatchValue('deq[0].a', '56789', '0', on_line=ref('second'))

