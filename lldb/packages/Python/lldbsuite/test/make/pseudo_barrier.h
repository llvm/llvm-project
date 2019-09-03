#include <atomic>
#include <thread>

typedef std::atomic<int> pseudo_barrier_t;

static inline void pseudo_barrier_wait(pseudo_barrier_t &barrier) {
  --barrier;
  while (barrier > 0)
    std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

static inline void pseudo_barrier_init(pseudo_barrier_t &barrier, int count) {
  barrier = count;
}
