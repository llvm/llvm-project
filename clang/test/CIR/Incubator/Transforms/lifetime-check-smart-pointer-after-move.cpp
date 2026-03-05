// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -fclangir-lifetime-check="history=invalid,null" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

namespace std {
template <typename T>
T&& move(T& t) {
  return static_cast<T&&>(t);
}

// Minimal unique_ptr implementation for testing
template <typename T>
class [[gsl::Owner(T)]] unique_ptr {
  T* ptr;

public:
  unique_ptr() : ptr(nullptr) {}
  explicit unique_ptr(T* p) : ptr(p) {}

  // Move constructor
  unique_ptr(unique_ptr&& other) : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  // Move assignment
  unique_ptr& operator=(unique_ptr&& other) {
    if (this != &other) {
      delete ptr;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  // Deleted copy operations
  unique_ptr(const unique_ptr&) = delete;
  unique_ptr& operator=(const unique_ptr&) = delete;

  ~unique_ptr() { delete ptr; }

  // Safe operations (allowed after move)
  T* get() const { return ptr; }
  T* release() { T* p = ptr; ptr = nullptr; return p; }
  void reset(T* p = nullptr) { delete ptr; ptr = p; }
  explicit operator bool() const { return ptr != nullptr; }

  // Unsafe operations (should warn if used after move)
  T& operator*() const { return *ptr; }
  T* operator->() const { return ptr; }
};

// Minimal shared_ptr implementation for testing
template <typename T>
class [[gsl::Owner(T)]] shared_ptr {
  T* ptr;

public:
  shared_ptr() : ptr(nullptr) {}
  explicit shared_ptr(T* p) : ptr(p) {}

  // Move constructor
  shared_ptr(shared_ptr&& other) : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  // Move assignment
  shared_ptr& operator=(shared_ptr&& other) {
    if (this != &other) {
      delete ptr;
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  ~shared_ptr() { delete ptr; }

  // Safe operations (allowed after move)
  T* get() const { return ptr; }
  void reset(T* p = nullptr) { delete ptr; ptr = p; }
  explicit operator bool() const { return ptr != nullptr; }

  // Unsafe operations (should warn if used after move)
  T& operator*() const { return *ptr; }
  T* operator->() const { return ptr; }
};

} // namespace std

struct Data {
  int value;
  void process();
};

// Test 1: Safe operations after move (unique_ptr)
void test_unique_ptr_safe_operations() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p);

  // Safe operations - should NOT warn
  int* raw = p.get();        // OK - get() is safe
  if (p) {                    // OK - bool conversion is safe
    // Not reached
  }
  p.reset();                  // OK - reset() is safe
}

// Test 2: Unsafe operations after move (unique_ptr)
void test_unique_ptr_unsafe_operations() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}

  // Unsafe operations - should warn
  int x = *p; // expected-warning {{use of invalid pointer 'p'}}
}

// Test 3: Unsafe arrow operator after move (unique_ptr)
void test_unique_ptr_arrow_after_move() {
  std::unique_ptr<Data> p(new Data());
  std::unique_ptr<Data> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}

  // Unsafe operation - should warn
  p->process(); // expected-warning {{use of invalid pointer 'p'}}
}

// Test 4: Reinit after move (unique_ptr)
// TODO: Implement reset() as reinitializing operation
/*
void test_unique_ptr_reinit() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p);

  p.reset(new int(10)); // Reinitialize
  int x = *p; // OK - reinitialized
}
*/

// Test 5: Safe operations after move (shared_ptr)
void test_shared_ptr_safe_operations() {
  std::shared_ptr<int> p(new int(42));
  std::shared_ptr<int> q = std::move(p);

  // Safe operations - should NOT warn
  int* raw = p.get();         // OK - get() is safe
  if (p) {                     // OK - bool conversion is safe
    // Not reached
  }
  p.reset();                   // OK - reset() is safe
}

// Test 6: Unsafe operations after move (shared_ptr)
void test_shared_ptr_unsafe_operations() {
  std::shared_ptr<int> p(new int(42));
  std::shared_ptr<int> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}

  // Unsafe operations - should warn
  int x = *p; // expected-warning {{use of invalid pointer 'p'}}
}

// Test 7: Unsafe arrow operator after move (shared_ptr)
void test_shared_ptr_arrow_after_move() {
  std::shared_ptr<Data> p(new Data());
  std::shared_ptr<Data> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}

  // Unsafe operation - should warn
  p->process(); // expected-warning {{use of invalid pointer 'p'}}
}

// Test 8: Move via function parameter (unique_ptr)
void consume_unique_ptr(std::unique_ptr<int>&& ptr) {}
void consume_two_unique_ptrs(std::unique_ptr<int>&& ptr1, std::unique_ptr<int>&& ptr2) {}

void test_unique_ptr_move_via_param() {
  std::unique_ptr<int> p(new int(42));
  consume_unique_ptr(std::move(p));

  // Safe after move
  if (p) {  // OK - bool conversion
    // Not reached
  }
}

// Test 9: Move via function parameter with unsafe use (unique_ptr)
void test_unique_ptr_move_param_unsafe() {
  std::unique_ptr<int> p(new int(42));
  consume_unique_ptr(std::move(p)); // expected-note {{moved here via std::move or rvalue reference}}

  // Unsafe after move
  int x = *p; // expected-warning {{use of invalid pointer 'p'}}
}

// Test 10: Multiple safe operations after move
void test_multiple_safe_ops() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p);

  // Multiple safe operations in sequence
  int* r1 = p.get();  // OK
  int* r2 = p.get();  // OK
  if (p) {}           // OK
  if (!p) {}          // OK
  p.reset();          // OK
}

// Test 11: Safe then unsafe
void test_safe_then_unsafe() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}

  int* raw = p.get();  // OK - safe operation
  int x = *p;           // expected-warning {{use of invalid pointer 'p'}}
}

// Test 12: Move in conditional
void test_move_in_conditional(bool cond) {
  std::unique_ptr<int> p(new int(42));
  if (cond) {
    std::unique_ptr<int> q = std::move(p); // expected-note {{moved here via std::move or rvalue reference}}
  }
  int* raw = p.get();  // OK - get() is safe even after conditional move
  int x = *p;           // expected-warning {{use of invalid pointer 'p'}}
}

// Test 13: Release after move
void test_release_after_move() {
  std::unique_ptr<int> p(new int(42));
  std::unique_ptr<int> q = std::move(p);

  int* raw = p.release(); // OK - release() is safe
}

// Test 14: Multiple owner arguments with rvalue references
// Regression test for emittedDiagnostics guard bug
void test_multi_arg_owner_move() {
  std::unique_ptr<int> x(new int(1));
  std::unique_ptr<int> y(new int(2));
  consume_unique_ptr(std::move(x)); // expected-note {{moved here via std::move or rvalue reference}}
  consume_two_unique_ptrs(std::move(x), std::move(y)); // expected-warning {{use of invalid pointer 'x'}}
                                                        // expected-note@-1 {{moved here via std::move or rvalue reference}}
  int use_y = *y; // expected-warning {{use of invalid pointer 'y'}}
}
