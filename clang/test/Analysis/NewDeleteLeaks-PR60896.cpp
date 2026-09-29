// RUN: %clang_analyze_cc1 -verify -analyzer-output=text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus \
// RUN:   -analyzer-checker=unix

#include "Inputs/system-header-simulator-for-malloc.h"

//===----------------------------------------------------------------------===//
// unique_ptr test cases 
//===----------------------------------------------------------------------===//
namespace unique_ptr_tests {

// Custom unique_ptr implementation for testing
template <typename T>
struct unique_ptr {
  T* ptr;
  unique_ptr(T* p) : ptr(p) {}
  ~unique_ptr() {
    // This destructor intentionally doesn't delete 'ptr' to validate that the
    // heuristic trusts that smart pointers (based on their class name) will
    // release the pointee even if it doesn't understand their destructor.
  }
  unique_ptr(unique_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
  T* get() const { return ptr; }
};

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
  return unique_ptr<T>(new T(args...));
}

// Test 1: Check that we report leaks for malloc when passing smart pointers
void add_unique_ptr(unique_ptr<int> ptr) {
  // The unique_ptr destructor will be called when ptr goes out of scope
}

void test_malloc_with_smart_ptr() {
  void *ptr = malloc(4); // expected-note {{Memory is allocated}}

  add_unique_ptr(make_unique<int>(1));
  (void)ptr;
  // expected-warning@+1 {{Potential leak of memory pointed to by 'ptr'}} expected-note@+1 {{Potential leak of memory pointed to by 'ptr'}}
}

// Test 2: Check that we don't report leaks for unique_ptr in temporary objects
struct Foo { 
  unique_ptr<int> i;
};

void add_foo(Foo foo) {
  // The unique_ptr destructor will be called when foo goes out of scope
}

void test_temporary_object() {
  // No warning should be emitted for this - the memory is managed by unique_ptr 
  // in the temporary Foo object, which will properly clean up the memory
  add_foo({make_unique<int>(1)});
}

// Test 3: Check that we don't report leaks for smart pointers in base class fields
struct Base {
  unique_ptr<int> base_ptr;
  Base() : base_ptr(nullptr) {}
  Base(unique_ptr<int>&& ptr) : base_ptr(static_cast<unique_ptr<int>&&>(ptr)) {}
};

struct Derived : public Base {
  int derived_field;
  Derived() : Base(), derived_field(0) {}
  Derived(unique_ptr<int>&& ptr, int field) : Base(static_cast<unique_ptr<int>&&>(ptr)), derived_field(field) {}
};

void add_derived(Derived derived) {
  // The unique_ptr destructor will be called when derived goes out of scope
  // This should include the base_ptr field from the base class
}

void test_base_class_smart_ptr() {
  // No warning should be emitted for this - the memory is managed by unique_ptr 
  // in the base class field of the temporary Derived object
  add_derived(Derived(make_unique<int>(1), 42));
}

// Test 4: Check that we don't report leaks for multiple owning arguments
struct SinglePtr {
  unique_ptr<int> ptr;
  SinglePtr(unique_ptr<int>&& p) : ptr(static_cast<unique_ptr<int>&&>(p)) {}
};

struct MultiPtr {
  unique_ptr<int> ptr1;
  unique_ptr<int> ptr2;
  unique_ptr<int> ptr3;
  
  MultiPtr(unique_ptr<int>&& p1, unique_ptr<int>&& p2, unique_ptr<int>&& p3)
    : ptr1(static_cast<unique_ptr<int>&&>(p1))
    , ptr2(static_cast<unique_ptr<int>&&>(p2))
    , ptr3(static_cast<unique_ptr<int>&&>(p3)) {}
};

void addMultiple(SinglePtr single, MultiPtr multi) {
  // All unique_ptr destructors will be called when the objects go out of scope
  // This tests handling of multiple by-value arguments with smart pointer fields
}

void test_multiple_owning_args() {
  // No warning should be emitted - all memory is properly managed by unique_ptr
  // in the temporary objects, which will properly clean up the memory
  addMultiple(
    SinglePtr(make_unique<int>(1)),
    MultiPtr(make_unique<int>(2), make_unique<int>(3), make_unique<int>(4))
  );
}

// Test 5: Check that we DO report leaks for raw pointers in mixed ownership scenarios
struct MixedOwnership {
  unique_ptr<int> smart_ptr;  // Should NOT leak (smart pointer managed)
  int *raw_ptr;               // Should leak (raw pointer)

  MixedOwnership() : smart_ptr(make_unique<int>(1)), raw_ptr(new int(42)) {} // expected-note {{Memory is allocated}}
};

void consume(MixedOwnership obj) {
  // The unique_ptr destructor will be called when obj goes out of scope
  // But raw_ptr will leak!
}

void test_mixed_ownership() {
  // This should report a leak for raw_ptr but not for smart_ptr
  consume(MixedOwnership()); // expected-note {{Calling default constructor for 'MixedOwnership'}} expected-note {{Returning from default constructor for 'MixedOwnership'}}
} // expected-warning {{Potential memory leak}} expected-note {{Potential memory leak}}

// Test 6: Check that we handle direct smart pointer constructor calls correctly
void test_direct_constructor() {
  // Direct constructor call - should not leak
  int* raw_ptr = new int(42);
  unique_ptr<int> smart(raw_ptr); // This should escape the raw_ptr symbol
  // No leak should be reported here since smart pointer takes ownership
}

void test_mixed_direct_constructor() {
  int* raw1 = new int(1); 
  int* raw2 = new int(2); // expected-note {{Memory is allocated}}
  
  unique_ptr<int> smart(raw1); // This should escape raw1
  // raw2 should leak since it's not managed by any smart pointer
  int x = *raw2; // expected-warning {{Potential leak of memory pointed to by 'raw2'}} expected-note {{Potential leak of memory pointed to by 'raw2'}}
}

// Test 7: Multiple memory owning arguments - demonstrates addTransition API usage
void addMultipleOwningArgs(
  unique_ptr<int> ptr1,
  unique_ptr<int> ptr2, 
  unique_ptr<int> ptr3
) {
  // All unique_ptr destructors will be called when arguments go out of scope
  // This tests handling of multiple smart pointer parameters in a single call
}

void test_multiple_memory_owning_arguments() {
  // No warning should be emitted - all memory is properly managed by unique_ptr
  // This test specifically exercises the addTransition API with multiple owning arguments
  addMultipleOwningArgs(
    make_unique<int>(1),
    make_unique<int>(2), 
    make_unique<int>(3)
  );
}

} // namespace unique_ptr_tests

//===----------------------------------------------------------------------===//
// Variadic constructor test cases
//===----------------------------------------------------------------------===//
namespace variadic_constructor_tests {

// Variadic constructor - test for potential out-of-bounds access
// This is the only test in this namespace and tests a scenario where Call.getNumArgs() > CD->getNumParams()
// We use a synthetic unique_ptr here to activate the specific logic in the MallocChecker that will test out of bounds
template <typename T>
struct unique_ptr {
  T* ptr;

  // Constructor with ellipsis - can receive more arguments than parameters  
  unique_ptr(T* p, ...) : ptr(p) {}

  ~unique_ptr() {
    // This destructor intentionally doesn't delete 'ptr' to validate that the
    // heuristic trusts that smart pointers (based on their class name) will
    // release the pointee even if it doesn't understand their destructor.
  }
};

void process_variadic_smart_ptr(unique_ptr<int> ptr) {
  // Function body doesn't matter for this test
}

void test_variadic_constructor_bounds() {
  void *malloc_ptr = malloc(4); // expected-note {{Memory is allocated}}
  
  // This call creates a smart pointer with more arguments than formal parameters
  // The constructor has 1 formal parameter (T* p) plus ellipsis, but we pass multiple args
  // This should trigger the bounds checking issue in handleSmartPointerConstructorArguments
  int* raw_ptr = new int(42);
  process_variadic_smart_ptr(unique_ptr<int>(raw_ptr, 1, 2, 3, 4, 5));
  
  (void)malloc_ptr;
} // expected-warning {{Potential leak of memory pointed to by 'malloc_ptr'}}
  // expected-note@-1 {{Potential leak of memory pointed to by 'malloc_ptr'}}

} // namespace variadic_constructor_tests

//===----------------------------------------------------------------------===//
// shared_ptr test cases
//===----------------------------------------------------------------------===//
namespace shared_ptr_tests {

// Custom shared_ptr implementation for testing
template <typename T>
struct shared_ptr {
  T* ptr;
  shared_ptr(T* p) : ptr(p) {}
  ~shared_ptr() {
    // This destructor intentionally doesn't delete 'ptr' to validate that the
    // heuristic trusts that smart pointers (based on their class name) will
    // release the pointee even if it doesn't understand their destructor.
  }
  shared_ptr(shared_ptr&& other) : ptr(other.ptr) { other.ptr = nullptr; }
  T* get() const { return ptr; }
};

template <typename T, typename... Args>
shared_ptr<T> make_shared(Args&&... args) {
  return shared_ptr<T>(new T(args...));
}

// Test 1: Check that we don't report leaks for shared_ptr in temporary objects
struct Foo { 
  shared_ptr<int> i;
};

void add_foo(Foo foo) {
  // The shared_ptr destructor will be called when foo goes out of scope
}

void test_temporary_object() {
  // No warning should be emitted for this - the memory is managed by shared_ptr 
  // in the temporary Foo object, which will properly clean up the memory
  add_foo({make_shared<int>(1)});
}

} // namespace shared_ptr_tests
