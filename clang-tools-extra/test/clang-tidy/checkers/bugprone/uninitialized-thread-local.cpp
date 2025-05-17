// RUN: %check_clang_tidy %s bugprone-uninitialized-thread-local %t

thread_local int global_tls = 1;

int main() {
  thread_local int local_tls = 2;
  {
    ++global_tls; // no warning
    ++local_tls; // no warning
  }
  auto f = []() {
    return local_tls + global_tls;
    // CHECK-MESSAGES: [[@LINE-1]]:12: warning: variable 'local_tls' might not have been initialized on the current thread.
  };
  return f();
}
