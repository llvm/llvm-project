// RUN: %clang_cc1 -std=gnu11 -fsyntax-only -verify=expected-nowarn %s
// RUN: %clang_cc1 -std=gnu11 -Wpointer-arith -fsyntax-only -verify=expected-warn %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify=expected-strict %s

// expected-nowarn-no-diagnostics

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __sized_by(f)  __attribute__((sized_by(f)))

//==============================================================================
// Test: counted_by on void* is allowed in GNU mode, rejected in strict mode
//==============================================================================

struct test_void_ptr_gnu {
  int count;
  // expected-warn-warning@+3{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+2{{use '__sized_by' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
  void* buf __counted_by(count);
};

struct test_const_void_ptr_gnu {
  int count;
  // expected-warn-warning@+3{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+2{{use '__sized_by' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'const void' is an incomplete type}}
  const void* buf __counted_by(count);
};

struct test_volatile_void_ptr_gnu {
  int count;
  // expected-warn-warning@+3{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+2{{use '__sized_by' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'volatile void' is an incomplete type}}
  volatile void* buf __counted_by(count);
};

struct test_const_volatile_void_ptr_gnu {
  int count;
  // expected-warn-warning@+3{{'counted_by' on a pointer to void is a GNU extension, treated as 'sized_by'}}
  // expected-warn-note@+2{{use '__sized_by' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by' cannot be applied to a pointer with pointee of unknown size because 'const volatile void' is an incomplete type}}
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
  // expected-warn-warning@+3{{'counted_by_or_null' on a pointer to void is a GNU extension, treated as 'sized_by_or_null'}}
  // expected-warn-note@+2{{use '__sized_by_or_null' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
  void* buf __counted_by_or_null(count);
};

struct test_const_void_ptr_or_null_gnu {
  int count;
  // expected-warn-warning@+3{{'counted_by_or_null' on a pointer to void is a GNU extension, treated as 'sized_by_or_null'}}
  // expected-warn-note@+2{{use '__sized_by_or_null' to suppress this warning}}
  // expected-strict-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'const void' is an incomplete type}}
  const void* buf __counted_by_or_null(count);
};
