// No warnings in regular compile
// RUN: %clang_cc1 -verify=no-tsan %s

// Emits warning with `-fsanitize=thread`
// RUN: %clang_cc1 -verify=with-tsan -fsanitize=thread %s

// No warnings if `-Wno-tsan` is passed
// RUN: %clang_cc1 -verify=no-tsan -fsanitize=thread -Wno-tsan %s

// Ignoring function
// RUN: echo "fun:main" > %t
// RUN: %clang_cc1 -verify=no-tsan -fsanitize=thread -fsanitize-ignorelist=%t %s

// Ignoring source file
// RUN: echo "src:%s" > %t
// RUN: %clang_cc1 -verify=no-tsan -fsanitize=thread -fsanitize-ignorelist=%t %s

// no-tsan-no-diagnostics

namespace std {
  enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst,
  };
  void atomic_thread_fence(memory_order) {}
};

__attribute__((no_sanitize("thread")))
void ignore_1() {
  std::atomic_thread_fence(std::memory_order_relaxed);
}

__attribute__((no_sanitize_thread))
void ignore_2() {
  std::atomic_thread_fence(std::memory_order_relaxed);
}

int main() {
  std::atomic_thread_fence(std::memory_order_relaxed); // with-tsan-warning {{'std::atomic_thread_fence' is not supported with '-fsanitize=thread'}}
}
