// RUN: %check_clang_tidy -std=c++20 %s bugprone-exception-escape %t -- -- -fexceptions

// Exercise cached diagnostics, catch contexts, recursion, and repeated calls.
void leaf() { throw 1; }

void catches() {
  try {
    leaf();
  } catch (int) {
  }
}

void escaped_after_catch() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'escaped_after_catch' which should not throw exceptions [bugprone-exception-escape]
  catches();
  leaf();
// CHECK-MESSAGES: :[[@LINE-1]]:3: note: frame #1: function 'escaped_after_catch' calls function 'leaf' here
}

void rethrow() {
  try {
    leaf();
  } catch (int) {
    throw;
  }
}

void escaped_rethrow() noexcept { rethrow(); }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'escaped_rethrow' which should not throw exceptions [bugprone-exception-escape]

void rethrows_current() { throw; }

void caught_contexts() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'caught_contexts' which should not throw exceptions [bugprone-exception-escape]
  try {
    throw 1;
  } catch (int) {
    try {
      rethrows_current();
    } catch (int) {
    }
  }
  try {
    throw 2.0;
  } catch (double) {
    rethrows_current();
  }
}

void cycle_a();
void cycle_b() { cycle_a(); }
void cycle_a() {
  cycle_b();
  throw 1;
}

void recursive_escape() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'recursive_escape' which should not throw exceptions [bugprone-exception-escape]
  try {
    cycle_a();
  } catch (int) {
  }
  cycle_b();
}

template <int N> void repeated() {
  repeated<N - 1>();
  repeated<N - 1>();
}
template <> void repeated<0>() { throw 1; }

void repeated_escape() noexcept { repeated<30>(); }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'repeated_escape' which should not throw exceptions [bugprone-exception-escape]

void all_caught() noexcept {
// CHECK-MESSAGES-NOT: warning: an exception may be thrown in function 'all_caught'
  try {
    repeated<30>();
  } catch (...) {
  }
}
