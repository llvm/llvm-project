// No warnings in regular compile
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=no-warnings %s

// Emits warnings with `-fsanitize=thread`
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=with-tsan,warn-global-function,warn-member-function -fsanitize=thread %s

// No warnings if `-Wno-tsan` is passed
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=no-warnings -fsanitize=thread -Wno-tsan %s

// Ignoring source file
// RUN: echo "src:*%{s:basename}" > %t
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=no-warnings -fsanitize=thread -fsanitize-ignorelist=%t %s

// Ignoring global function
// RUN: echo "fun:*global_function_to_maybe_ignore*" > %t
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=with-tsan,warn-member-function -fsanitize=thread -fsanitize-ignorelist=%t %s

// Ignoring "S::member_function"
// RUN: echo "fun:_ZN1S15member_functionEv" > %t
// RUN: %clang_cc1 -triple=x86_64-linux-unknown -verify=with-tsan,warn-global-function -fsanitize=thread -fsanitize-ignorelist=%t %s

// no-warnings-no-diagnostics

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

void global_function_to_maybe_ignore() {
  std::atomic_thread_fence(std::memory_order_relaxed); // warn-global-function-warning {{'std::atomic_thread_fence' is not supported with '-fsanitize=thread'}}
}

__attribute__((no_sanitize("thread")))
void no_sanitize_1() {
  std::atomic_thread_fence(std::memory_order_relaxed);

  auto lam = []() {
    std::atomic_thread_fence(std::memory_order_relaxed); // with-tsan-warning {{'std::atomic_thread_fence' is not supported with '-fsanitize=thread'}}
  };
}

__attribute__((no_sanitize_thread))
void no_sanitize_2() {
  std::atomic_thread_fence(std::memory_order_relaxed);
}

__attribute__((disable_sanitizer_instrumentation))
void no_sanitize_3() {
  std::atomic_thread_fence(std::memory_order_relaxed);
}

void no_sanitize_lambda() {
  auto lam = [] () __attribute__((no_sanitize("thread"))) {
    std::atomic_thread_fence(std::memory_order_relaxed);
  };
}

struct S {
public:
  void member_function() {
    std::atomic_thread_fence(std::memory_order_relaxed); // warn-member-function-warning {{'std::atomic_thread_fence' is not supported with '-fsanitize=thread'}}
  }
};
