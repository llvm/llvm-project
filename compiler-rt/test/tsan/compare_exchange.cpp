// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %deflake %env_tsan_opts=atexit_sleep_ms=50 %run %t 2>&1 | FileCheck --check-prefix=CHECK-REPORT %s

#include <atomic>
#include <cassert>
#include <stdio.h>
#include <thread>

#define NUM_ORDS 16
#define NUM_THREADS NUM_ORDS * 2
struct node {
  int val;
};
std::atomic<node *> _nodes[NUM_THREADS] = {};

void f1(int i) {
  auto n = new node();
  n->val = 42;
  _nodes[i].store(n, std::memory_order_release);
}

template <int version>
void f2(int i, std::memory_order mo, std::memory_order fmo) {
  node *expected = nullptr;
  while (expected == nullptr) {
    _nodes[i].compare_exchange_weak(expected, nullptr, mo, fmo);
  };

  ++expected->val;
  assert(expected->val == 43);
}

struct MemOrdSuccFail {
  std::memory_order mo;
  std::memory_order fmo;
};

MemOrdSuccFail OrdList[NUM_ORDS] = {
    {std::memory_order_release, std::memory_order_relaxed},
    {std::memory_order_release, std::memory_order_acquire},
    {std::memory_order_release, std::memory_order_consume},
    {std::memory_order_release, std::memory_order_seq_cst},

    {std::memory_order_acq_rel, std::memory_order_relaxed},
    {std::memory_order_acq_rel, std::memory_order_acquire},
    {std::memory_order_acq_rel, std::memory_order_consume},
    {std::memory_order_acq_rel, std::memory_order_seq_cst},

    {std::memory_order_seq_cst, std::memory_order_relaxed},
    {std::memory_order_seq_cst, std::memory_order_acquire},
    {std::memory_order_seq_cst, std::memory_order_consume},
    {std::memory_order_seq_cst, std::memory_order_seq_cst},

    {std::memory_order_relaxed, std::memory_order_relaxed},
    {std::memory_order_relaxed, std::memory_order_acquire},
    {std::memory_order_relaxed, std::memory_order_consume},
    {std::memory_order_relaxed, std::memory_order_seq_cst},
};

int main() {
  std::thread threads[NUM_THREADS];
  int ords = 0;

  // Instantiate a new f2 for each MO so we can dedup reports and actually
  // make sure relaxed FMO triggers a warning for every different MO.
  for (unsigned t = 0; t < 8; t += 2) {
    threads[t] = std::thread(f1, t);
    threads[t + 1] = std::thread(f2<0>, t, OrdList[ords].mo, OrdList[ords].fmo);
    threads[t].join();
    threads[t + 1].join();
    ords++;
  }

  for (unsigned t = 8; t < 16; t += 2) {
    threads[t] = std::thread(f1, t);
    threads[t + 1] = std::thread(f2<1>, t, OrdList[ords].mo, OrdList[ords].fmo);
    threads[t].join();
    threads[t + 1].join();
    ords++;
  }

  for (unsigned t = 16; t < 24; t += 2) {
    threads[t] = std::thread(f1, t);
    threads[t + 1] = std::thread(f2<2>, t, OrdList[ords].mo, OrdList[ords].fmo);
    threads[t].join();
    threads[t + 1].join();
    ords++;
  }

  for (unsigned t = 24; t < 32; t += 2) {
    threads[t] = std::thread(f1, t);
    threads[t + 1] = std::thread(f2<3>, t, OrdList[ords].mo, OrdList[ords].fmo);
    threads[t].join();
    threads[t + 1].join();
    ords++;
  }

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-REPORT: WARNING: ThreadSanitizer: data race
// CHECK-REPORT: WARNING: ThreadSanitizer: data race
// CHECK-REPORT: WARNING: ThreadSanitizer: data race
// CHECK-REPORT: WARNING: ThreadSanitizer: data race
// CHECK-REPORT: DONE
// CHECK-REPORT: ThreadSanitizer: reported 4 warnings
