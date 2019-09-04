#include <atomic>
#include <thread>

typedef std::atomic<int> pseudo_barrier_t;

static inline void pseudo_barrier_wait(pseudo_barrier_t &barrier) {
  --barrier;
  while (barrier > 0)
    std::this_thread::yield();
}

static inline void pseudo_barrier_init(pseudo_barrier_t &barrier, int count) {
  barrier = count;
}
