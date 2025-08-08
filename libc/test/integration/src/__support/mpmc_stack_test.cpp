#include "src/__support/CPP/atomic.h"
#include "src/__support/mpmc_stack.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

void smoke_test() {
  MPMCStack<int> stack;
  for (int i = 0; i <= 100; ++i)
    if (!stack.push(i))
      __builtin_trap();
  for (int i = 100; i >= 0; --i)
    if (*stack.pop() != i)
      __builtin_trap();
  if (stack.pop())
    __builtin_trap(); // Should be empty now.
}

void multithread_test() {
  constexpr static size_t NUM_THREADS = 5;
  constexpr static size_t NUM_PUSHES = 100;
  struct State {
    MPMCStack<size_t> stack;
    cpp::Atomic<size_t> counter = 0;
    cpp::Atomic<bool> flags[NUM_PUSHES];
  } state;
  pthread_t threads[NUM_THREADS];
  for (size_t i = 0; i < NUM_THREADS; ++i) {
    LIBC_NAMESPACE::pthread_create(
        &threads[i], nullptr,
        [](void *arg) -> void * {
          State *state = static_cast<State *>(arg);
          for (;;) {
            size_t current = state->counter.fetch_add(1);
            if (current >= NUM_PUSHES)
              break;
            if (!state->stack.push(current))
              __builtin_trap();
          }
          while (auto res = state->stack.pop())
            state->flags[res.value()].store(true);
          return nullptr;
        },
        &state);
  }
  for (pthread_t thread : threads)
    LIBC_NAMESPACE::pthread_join(thread, nullptr);
  while (cpp::optional<size_t> res = state.stack.pop())
    state.flags[res.value()].store(true);
  for (size_t i = 0; i < NUM_PUSHES; ++i)
    if (!state.flags[i].load())
      __builtin_trap();
}

void multithread_push_all_test() {
  constexpr static size_t NUM_THREADS = 4;
  constexpr static size_t BATCH_SIZE = 10;
  constexpr static size_t NUM_BATCHES = 20;
  struct State {
    MPMCStack<size_t> stack;
    cpp::Atomic<size_t> counter = 0;
    cpp::Atomic<bool> flags[NUM_THREADS * BATCH_SIZE * NUM_BATCHES];
  } state;
  pthread_t threads[NUM_THREADS];

  for (size_t i = 0; i < NUM_THREADS; ++i) {
    LIBC_NAMESPACE::pthread_create(
        &threads[i], nullptr,
        [](void *arg) -> void * {
          State *state = static_cast<State *>(arg);
          size_t values[BATCH_SIZE];

          for (size_t batch = 0; batch < NUM_BATCHES; ++batch) {
            // Prepare batch of values
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
              size_t current = state->counter.fetch_add(1);
              values[j] = current;
            }

            // Push all values in batch
            if (!state->stack.push_all(values, BATCH_SIZE))
              __builtin_trap();
          }

          // Pop and mark all values
          while (auto res = state->stack.pop()) {
            size_t value = res.value();
            if (value < NUM_THREADS * BATCH_SIZE * NUM_BATCHES)
              state->flags[value].store(true);
          }
          return nullptr;
        },
        &state);
  }

  for (pthread_t thread : threads)
    LIBC_NAMESPACE::pthread_join(thread, nullptr);

  // Pop any remaining values
  while (cpp::optional<size_t> res = state.stack.pop()) {
    size_t value = res.value();
    if (value < NUM_THREADS * BATCH_SIZE * NUM_BATCHES)
      state.flags[value].store(true);
  }

  // Verify all values were processed
  for (size_t i = 0; i < NUM_THREADS * BATCH_SIZE * NUM_BATCHES; ++i)
    if (!state.flags[i].load())
      __builtin_trap();
}

TEST_MAIN() {
  smoke_test();
  multithread_test();
  multithread_push_all_test();
  return 0;
}
