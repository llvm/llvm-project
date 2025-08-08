// RUN: %clang_analyze_cc1 -verify -analyzer-output=text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus \
// RUN:   -analyzer-checker=unix
// expected-no-diagnostics

#include "Inputs/system-header-simulator-for-malloc.h"

//===----------------------------------------------------------------------===//
// Check that we don't report leaks for unique_ptr in temporary objects
//===----------------------------------------------------------------------===//
namespace unique_ptr_temporary_PR60896 {

// We use a custom implementation of unique_ptr for testing purposes
template <typename T>
struct unique_ptr {
  T* ptr;
  unique_ptr(T* p) : ptr(p) {}
  ~unique_ptr() { delete ptr; }
  unique_ptr(unique_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
  T* get() const { return ptr; }
};

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>(new T(args...));
}

// The test case that demonstrates the issue
struct Foo { 
  unique_ptr<int> i;
};

void add(Foo foo) {
  // The unique_ptr destructor will be called when foo goes out of scope
}

void test() {
  // No warning should be emitted for this - the memory is managed by unique_ptr 
  // in the temporary Foo object, which will properly clean up the memory
  add({make_unique<int>(1)});
}

} // namespace unique_ptr_temporary_PR60896
