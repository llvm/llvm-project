// RUN: %clang_cc1 -fsyntax-only -verify=expected-nowarn %s
// RUN: %clang_cc1 -Wpointer-arith -fsyntax-only -verify=expected-warn %s
// RUN: %clang_cc1 -fexperimental-bounds-safety -fsyntax-only -verify=expected-bounds %s

// expected-nowarn-no-diagnostics
// expected-bounds-no-diagnostics

#define NULL (void*)0
#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

//==============================================================================
// Test: counted_by on void* is allowed (warns with -Wpointer-arith)
//==============================================================================

struct test_void_ptr_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+1{{use '__sized_by' to suppress this warning}}
  void* buf __counted_by(count);
};

struct test_const_void_ptr_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+1{{use '__sized_by' to suppress this warning}}
  const void* buf __counted_by(count);
};

struct test_volatile_void_ptr_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+1{{use '__sized_by' to suppress this warning}}
  volatile void* buf __counted_by(count);
};

struct test_const_volatile_void_ptr_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+1{{use '__sized_by' to suppress this warning}}
  const volatile void* buf __counted_by(count);
};

// Verify sized_by still works the same way (always allowed, no warning)
struct test_sized_by_void_ptr {
  int size;
  void* buf __sized_by(size);  // OK in both modes, no warning
};

//==============================================================================
// Test: counted_by_or_null on void* behaves the same
//==============================================================================

struct test_void_ptr_or_null_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by_or_null' on a pointer to void is a GNU extension, treated as 'sized_by_or_null'}}
  // expected-warn-note@+1{{use '__sized_by_or_null' to suppress this warning}}
  void* buf __counted_by_or_null(count);
};

struct test_const_void_ptr_or_null_gnu {
  int count;
  // expected-warn-warning@+2{{'counted_by_or_null' on a pointer to void is a GNU extension, treated as 'sized_by_or_null'}}
  // expected-warn-note@+1{{use '__sized_by_or_null' to suppress this warning}}
  const void* buf __counted_by_or_null(count);
};

//==============================================================================
// Test: Using void* __counted_by(...) pointers (not just declaring them)
//==============================================================================

// Verify that void* __counted_by pointers can be used as rvalues, assigned to,
// passed to functions, etc.

void* use_as_rvalue(struct test_void_ptr_gnu* t) {
  return t->buf;
}

void assign_to_pointer(struct test_void_ptr_gnu* t) {
  t->buf = NULL;
  t->count = 0;
}

extern void* my_allocator(unsigned long);

void assign_from_allocator(struct test_void_ptr_gnu* t) {
  t->buf = my_allocator(100);
  t->count = 100;
}

void takes_void_ptr(void* p);

void pass_to_function(struct test_void_ptr_gnu* t) {
  takes_void_ptr(t->buf);
}

void* pointer_arithmetic(struct test_void_ptr_gnu* t) {
  // expected-warn-warning@+1{{arithmetic on a pointer to void is a GNU extension}}
  return t->buf + 10;
}
