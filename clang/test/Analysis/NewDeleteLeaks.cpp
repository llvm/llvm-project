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

#include "Inputs/system-header-simulator-for-malloc.h"

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
