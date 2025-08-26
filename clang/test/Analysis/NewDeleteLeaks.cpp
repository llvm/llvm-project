// RUN: %clang_analyze_cc1 -verify -analyzer-output=text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-config \
// RUN:     unix.DynamicMemoryModeling:AddNoOwnershipChangeNotes=false

// RUN: %clang_analyze_cc1 -verify=expected,ownership -analyzer-output=text %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-config \
// RUN:     unix.DynamicMemoryModeling:AddNoOwnershipChangeNotes=true
// RUN: %clang_analyze_cc1 -std=c++20 -analyzer-checker=cplusplus.NewDeleteLeaks -verify %s

#include "Inputs/system-header-simulator-for-malloc.h"

// Minimal move, no headers needed, C++11+
namespace nstd {

template <class T>
struct remove_reference { using type = T; };
template <class T>
struct remove_reference<T&> { using type = T; };
template <class T>
struct remove_reference<T&&> { using type = T; };

template <class T>
constexpr typename remove_reference<T>::type&& move(T&& t) noexcept {
    using U = typename remove_reference<T>::type;
    return static_cast<U&&>(t);
}

} // namespace nstd



//===----------------------------------------------------------------------===//
// Report for which we expect NoOwnershipChangeVisitor to add a new note.
//===----------------------------------------------------------------------===//

bool coin();

// TODO: AST analysis of sink would reveal that it doesn't intent to free the
// allocated memory, but in this instance, its also the only function with
// the ability to do so, we should see a note here.
namespace memory_allocated_in_fn_call {

void sink(int *P) {
}

void foo() {
  sink(new int(5)); // expected-note {{Memory is allocated}}
} // expected-warning {{Potential memory leak [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential memory leak}}

} // namespace memory_allocated_in_fn_call

// Realize that sink() intends to deallocate memory, assume that it should've
// taken care of the leaked object as well.
namespace memory_passed_to_fn_call_delete {

void sink(int *P) {
  if (coin()) // ownership-note {{Assuming the condition is false}}
              // ownership-note@-1 {{Taking false branch}}
    delete P;
} // ownership-note {{Returning without deallocating memory or storing the pointer for later deallocation}}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  sink(ptr);             // ownership-note {{Calling 'sink'}}
                         // ownership-note@-1 {{Returning from 'sink'}}
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_passed_to_fn_call_delete

namespace memory_passed_to_fn_call_free {

void sink(int *P) {
  if (coin()) // ownership-note {{Assuming the condition is false}}
              // ownership-note@-1 {{Taking false branch}}
    free(P);
} // ownership-note {{Returning without deallocating memory or storing the pointer for later deallocation}}

void foo() {
  int *ptr = (int *)malloc(sizeof(int)); // expected-note {{Memory is allocated}}
  sink(ptr);                             // ownership-note {{Calling 'sink'}}
                                         // ownership-note@-1 {{Returning from 'sink'}}
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [unix.Malloc]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_passed_to_fn_call_free

// Function pointers cannot be resolved syntactically.
namespace memory_passed_to_fn_call_free_through_fn_ptr {
void (*freeFn)(void *) = free;

void sink(int *P) {
  if (coin())
    freeFn(P);
}

void foo() {
  int *ptr = (int *)malloc(sizeof(int)); // expected-note {{Memory is allocated}}
  sink(ptr);
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [unix.Malloc]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_passed_to_fn_call_free_through_fn_ptr

namespace memory_shared_with_ptr_of_shorter_lifetime {

void sink(int *P) {
  int *Q = P;
  if (coin()) // ownership-note {{Assuming the condition is false}}
              // ownership-note@-1 {{Taking false branch}}
    delete P;
  (void)Q;
} // ownership-note {{Returning without deallocating memory or storing the pointer for later deallocation}}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  sink(ptr);             // ownership-note {{Calling 'sink'}}
                         // ownership-note@-1 {{Returning from 'sink'}}
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_shared_with_ptr_of_shorter_lifetime

//===----------------------------------------------------------------------===//
// Report for which we *do not* expect NoOwnershipChangeVisitor add a new note,
// nor do we want it to.
//===----------------------------------------------------------------------===//

namespace memory_not_passed_to_fn_call {

void sink(int *P) {
  if (coin())
    delete P;
}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  int *q = nullptr;
  sink(q);
  (void)ptr;
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_not_passed_to_fn_call

namespace memory_shared_with_ptr_of_same_lifetime {

void sink(int *P, int **Q) {
  // NOTE: Not a job of NoOwnershipChangeVisitor, but maybe this could be
  // highlighted still?
  *Q = P;
}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  int *q = nullptr;
  sink(ptr, &q);
} // expected-warning {{Potential leak of memory pointed to by 'q' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_shared_with_ptr_of_same_lifetime

namespace memory_passed_into_fn_that_doesnt_intend_to_free {

void sink(int *P) {
}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  sink(ptr);
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_passed_into_fn_that_doesnt_intend_to_free

namespace memory_passed_into_fn_that_doesnt_intend_to_free2 {

void bar();

void sink(int *P) {
  // Correctly realize that calling bar() doesn't mean that this function would
  // like to deallocate anything.
  bar();
}

void foo() {
  int *ptr = new int(5); // expected-note {{Memory is allocated}}
  sink(ptr);
} // expected-warning {{Potential leak of memory pointed to by 'ptr' [cplusplus.NewDeleteLeaks]}}
// expected-note@-1 {{Potential leak}}

} // namespace memory_passed_into_fn_that_doesnt_intend_to_free2

namespace refkind_from_unoallocated_to_allocated {

// RefKind of the symbol changed from nothing to Allocated. We don't want to
// emit notes when the RefKind changes in the stack frame.
static char *malloc_wrapper_ret() {
  return (char *)malloc(12); // expected-note {{Memory is allocated}}
}
void use_ret() {
  char *v;
  v = malloc_wrapper_ret(); // expected-note {{Calling 'malloc_wrapper_ret'}}
                            // expected-note@-1 {{Returned allocated memory}}
} // expected-warning {{Potential leak of memory pointed to by 'v' [unix.Malloc]}}
// expected-note@-1 {{Potential leak of memory pointed to by 'v'}}

} // namespace refkind_from_unoallocated_to_allocated

// Check that memory leak is reported against a symbol if the last place it's
// mentioned is a base region of a lazy compound value, as the program cannot
// possibly free that memory.
namespace symbol_reaper_lifetime {
struct Nested {
  int buf[2];
};
struct Wrapping {
  Nested data;
};

Nested allocateWrappingAndReturnNested() {
  // expected-note@+1 {{Memory is allocated}}
  Wrapping const* p = new Wrapping();
  // expected-warning@+2 {{Potential leak of memory pointed to by 'p'}}
  // expected-note@+1    {{Potential leak of memory pointed to by 'p'}}
  return p->data;
}

void caller() {
  // expected-note@+1 {{Calling 'allocateWrappingAndReturnNested'}}
  Nested n = allocateWrappingAndReturnNested();
  (void)n;
} // no-warning: No potential memory leak here, because that's been already reported.
} // namespace symbol_reaper_lifetime


// Minimal RAII class that properly deletes its pointer.
class Bar {
public:
  explicit Bar(int *ptr) : ptr_(ptr) {}
  ~Bar() {
    if (ptr_) {
      delete ptr_;
      ptr_ = nullptr;
    }
  }

  Bar(const Bar &) = delete;
  Bar &operator=(const Bar &) = delete;

  Bar(Bar &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Bar &operator=(Bar &&other) noexcept {
    if (this != &other) {
      delete ptr_;
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  int operator*() const { return *ptr_; }

private:
  int *ptr_;
};

// Factory returning a prvalue Bar that owns a freshly allocated int.
static Bar make_bar(int v) { return Bar(new int(v)); }

struct Foo {
  Bar a;
  Bar b;
};

struct FooWithConstructor {
  Bar a;
  Bar b;
  FooWithConstructor(Bar &&original_a, Bar &&original_b)
      : a(nstd::move(original_a)), b(nstd::move(original_b)) {}
};

//===----------------------------------------------------------------------===//
// No-false-positive regression tests: these must be silent
//===----------------------------------------------------------------------===//

namespace prvalue_aggregate_transfer {

void ok_aggregate_from_factory() {
  Foo foo = {make_bar(1), make_bar(2)}; // expected-no-diagnostics
}

void ok_aggregate_from_temporary_exprs() {
  Foo foo = {Bar(new int(1)), Bar(new int(2))}; // expected-no-diagnostics
}

void ok_ctor_from_factory_rvalues() {
  FooWithConstructor foo = {make_bar(1), make_bar(2)}; // expected-no-diagnostics
}

} // namespace prvalue_aggregate_transfer

//===----------------------------------------------------------------------===//
// True-positive regression tests: these should still warn
//===----------------------------------------------------------------------===//

class BarNoDelete {
public:
  explicit BarNoDelete(int *ptr) : ptr_(ptr) {}
  ~BarNoDelete() {} // intentionally missing delete -> leak

  BarNoDelete(const BarNoDelete &) = delete;
  BarNoDelete &operator=(const BarNoDelete &) = delete;

  BarNoDelete(BarNoDelete &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  BarNoDelete &operator=(BarNoDelete &&other) noexcept {
    if (this != &other) {
      // no delete of old ptr_ on purpose
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

private:
  int *ptr_;
};

static BarNoDelete make_bar_nd(int v) { return BarNoDelete(new int(v)); }

struct FooND {
  BarNoDelete a;
  BarNoDelete b;
};

namespace prvalue_aggregate_positive {

void leak_aggregate_from_factory() {
  FooND f = {make_bar_nd(1), make_bar_nd(2)};
  // expected-warning@-1 {{Potential memory leak}}
}

void leak_direct_member() {
  BarNoDelete b(new int(3));
  // expected-warning@-1 {{Potential memory leak}}
}

} // namespace prvalue_aggregate_positive

//===----------------------------------------------------------------------===//
// Guard tests: neighboring behaviors that must remain intact
// These ensure we didn't weaken unrelated diagnostics (mismatch/double-delete).
//===----------------------------------------------------------------------===//

namespace guards {

void mismatch_array_delete() {
  int *p = new int[4];
  delete p; // expected-warning {{mismatched deallocation: 'delete' should be 'delete[]'}}
}

void double_delete() {
  int *p = new int(1);
  delete p;
  delete p; // expected-warning {{Attempt to free released memory}}
}

} // namespace guards
