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

// Custom unique_ptr implementation for testing
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

//===----------------------------------------------------------------------===//
// Check that we don't report leaks for shared_ptr in temporary objects
//===----------------------------------------------------------------------===//
namespace shared_ptr_temporary_PR60896 {

// Custom shared_ptr implementation for testing
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

} // namespace shared_ptr_temporary_PR60896

//===----------------------------------------------------------------------===//
// Check that we don't report leaks for smart pointers in base class fields
//===----------------------------------------------------------------------===//
namespace base_class_smart_ptr_PR60896 {

// Custom unique_ptr implementation for testing
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

// Base class with smart pointer field
struct Base {
  unique_ptr<int> base_ptr;
  Base() : base_ptr(nullptr) {}
  Base(unique_ptr<int>&& ptr) : base_ptr(static_cast<unique_ptr<int>&&>(ptr)) {}
};

// Derived class that inherits the smart pointer field
struct Derived : public Base {
  int derived_field;
  Derived() : Base(), derived_field(0) {}
  Derived(unique_ptr<int>&& ptr, int field) : Base(static_cast<unique_ptr<int>&&>(ptr)), derived_field(field) {}
};

void add(Derived derived) {
  // The unique_ptr destructor will be called when derived goes out of scope
  // This should include the base_ptr field from the base class
}

void test() {
  // No warning should be emitted for this - the memory is managed by unique_ptr 
  // in the base class field of the temporary Derived object
  add(Derived(make_unique<int>(1), 42));
}

} // namespace base_class_smart_ptr_PR60896
