// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus,unix -verify %s
// expected-no-diagnostics

#include "Inputs/system-header-simulator-for-malloc.h"

// Test shared_ptr support in the same pattern as the original PR60896 test
namespace shared_ptr_test {

template <typename T>
struct shared_ptr {
  T* ptr;
  shared_ptr(T* p) : ptr(p) {}
  ~shared_ptr() { delete ptr; }
  shared_ptr(shared_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
  T* get() const { return ptr; }
};

template <typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args) {
  return shared_ptr<T>(new T(args...));
}

struct Foo { 
  shared_ptr<int> i;
};

void add(Foo foo) {
  // The shared_ptr destructor will be called when foo goes out of scope
}

void test() {
  // No warning should be emitted for this - the memory is managed by shared_ptr 
  // in the temporary Foo object, which will properly clean up the memory
  add({make_shared<int>(1)});
}

} // namespace shared_ptr_test 