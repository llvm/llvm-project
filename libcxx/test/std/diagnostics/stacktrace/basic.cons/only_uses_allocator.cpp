//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;   [1]
  static basic_stacktrace current(size_type skip,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [2]
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [3]

  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);            [4]
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;                            [5]

  basic_stacktrace(const basic_stacktrace& other);                                            [6]
  basic_stacktrace(basic_stacktrace&& other) noexcept;                                        [7]
  basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);               [8]
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                    [9]
  basic_stacktrace& operator=(const basic_stacktrace& other);                                 [10]
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                                   [11]

  ~basic_stacktrace();                                                                        [12]
*/

#include <cassert>
#include <iostream>
#include <stacktrace>

/**
 * This file includes tests which ensure any allocations performed by `basic_stacktrace`
 * are done via the user-provided allocator, and not via bare `malloc` / `operator new`s.
 * Our overridden `malloc` below will fail if `malloc_disabled` is `true`.
 * The program's startup code may need to malloc, so we'll allow malloc initially.
 * This is only activated during the "test_no_malloc_or_new_ex_allocator" test,
 * during which it should be using a `test_alloc` we'll provide (which knows how to
 * unblock `malloc` temporarily).
 */
bool malloc_disabled{false};

void* malloc(size_t size) {
  // If flag has not been temporarily disabled by our allocator, abort
  if (malloc_disabled) {
    abort();
  }
  // Since we overrode malloc with this function, and there's no way to call up
  // to the "real malloc", allocate a different way.  Assumes nothing actually uses `calloc`.
  return calloc(1, size);
}

// All the various ways to monkey around with heap memory without an allocator:
// we'll simply redirect these through our gate-keeping `malloc` above.
void* operator new(size_t size) { return malloc(size); }
void* operator new[](size_t size) { return malloc(size); }
void operator delete(void* ptr) noexcept { free(ptr); }
void operator delete(void* ptr, size_t) noexcept { free(ptr); }
void operator delete[](void* ptr) noexcept { free(ptr); }
void operator delete[](void* ptr, size_t) noexcept { free(ptr); }

/** RAII-style scope object to temporarily permit heap allocations, used by `test_alloc`.*/
struct scope_enable_malloc {
  bool prev_malloc_disabled_;
  scope_enable_malloc() : prev_malloc_disabled_(malloc_disabled) { malloc_disabled = false; }
  ~scope_enable_malloc() { malloc_disabled = true; }
};

template <typename T>
struct test_alloc {
  using size_type     = size_t;
  using value_type    = T;
  using pointer       = T*;
  using const_pointer = T const*;

  template <typename U>
  struct rebind {
    using other = test_alloc<U>;
  };

  std::allocator<T> wrapped_{};

  test_alloc() = default;

  template <typename U>
  test_alloc(test_alloc<U> const& rhs) : wrapped_(rhs.wrapped_) {}

  bool operator==(auto const& rhs) const { return &rhs == this; }
  bool operator==(test_alloc const&) const { return true; }

  T* allocate(size_t n) { return wrapped_.allocate(n); }

  auto allocate_at_least(size_t n) { return wrapped_.allocate_at_least(n); }

  void deallocate(T* ptr, size_t n) { return wrapped_.deallocate(ptr, n); }
};

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;   [1]
  static basic_stacktrace current(size_type skip,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [2]
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [3]

  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);            [4]
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;                            [5]

  basic_stacktrace(const basic_stacktrace& other);                                            [6]
  basic_stacktrace(basic_stacktrace&& other) noexcept;                                        [7]
  basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);               [8]
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                    [9]
  basic_stacktrace& operator=(const basic_stacktrace& other);                                 [10]
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                                   [11]

  ~basic_stacktrace();                                                                        [12]
*/

// clang-format off
uint32_t test1_line;
uint32_t test2_line;

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE
std::basic_stacktrace<A> test1(A& alloc) {
  test1_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret = std::basic_stacktrace<A>::current(alloc);
  return ret;
}

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE
std::basic_stacktrace<A> test2(A& alloc) {
  test2_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret = test1(alloc);
  return ret;
}
// clang-format on

/*
    [1]
    static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;

    Returns: A basic_stacktrace object with frames_ storing the stacktrace of the current evaluation
    in the current thread of execution, or an empty basic_stacktrace object if the initialization of
    frames_ failed. alloc is passed to the constructor of the frames_ object.
    [Note 1: If the stacktrace was successfully obtained, then frames_.front() is the stacktrace_entry
    representing approximately the current evaluation, and frames_.back() is the stacktrace_entry
    representing approximately the initial function of the current thread of execution. - end note]
  */
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_current() {
  test_alloc<std::stacktrace_entry> alloc;
  uint32_t main_line = __LINE__ + 1;
  auto st            = test2(alloc);

  std::cerr << "*** Stacktrace obtained at line " << main_line << '\n' << st << '\n';

  assert(st.size() >= 3);
  assert(st[0]);
  assert(st[0].native_handle());
  assert(st[0].description().contains("test1"));
  assert(st[0].source_file().contains("basic.cons.pass.cpp"));
  assert(st[1]);
  assert(st[1].native_handle());
  assert(st[1].description().contains("test2"));
  assert(st[1].source_file().contains("basic.cons.pass.cpp"));
  assert(st[2]);
  assert(st[2].native_handle());
  assert(st[2].description().contains("test_current"));
  assert(st[2].source_file().contains("basic.cons.pass.cpp"));

  // We unfortunately cannot guarantee the following; in CI, and possibly on users' build machines,
  // there may not be an up-to-date version of e.g. `addr2line`.
  // assert(st[0].source_file().ends_with("basic.cons.pass.cpp"));
  // assert(st[0].source_line() == test1_line);
  // assert(st[1].source_file().ends_with("basic.cons.pass.cpp"));
  // assert(st[1].source_line() == test2_line);
  // assert(st[2].source_file().ends_with("basic.cons.pass.cpp"));
  // assert(st[2].source_line() == main_line);
}

/*
  [2]
  static basic_stacktrace current(size_type skip,
                              const allocator_type& alloc = allocator_type()) noexcept;
  Let t be a stacktrace as-if obtained via basic_stacktrace::current(alloc). Let n be t.size().
  Returns: A basic_stacktrace object where frames_ is direct-non-list-initialized from arguments
  t.begin() + min(n, skip), t.end(), and alloc, or an empty basic_stacktrace object if the
  initialization of frames_ failed.
*/
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_current_with_skip() {
  // Use default allocator for simplicity; alloc is covered above
  auto st_skip0 = std::stacktrace::current();
  std::cerr << "*** st_skip0:\n" << st_skip0 << '\n';
  assert(st_skip0.size() >= 2);
  auto st_skip1 = std::stacktrace::current(1);
  std::cerr << "*** st_skip1:\n" << st_skip1 << '\n';
  assert(st_skip1.size() >= 1);
  assert(st_skip0.size() == st_skip1.size() + 1);
  assert(st_skip0[1] == st_skip1[0]);
  auto st_skip_many = std::stacktrace::current(1 << 20);
  assert(st_skip_many.empty());
}

/*
  [3]
  static basic_stacktrace current(size_type skip, size_type max_depth,
                              const allocator_type& alloc = allocator_type()) noexcept;
  Let t be a stacktrace as-if obtained via basic_stacktrace::current(alloc). Let n be t.size().
  Preconditions: skip <= skip + max_depth is true.
  Returns: A basic_stacktrace object where frames_ is direct-non-list-initialized from arguments
  t.begin() + min(n, skip), t.begin() + min(n, skip + max_depth), and alloc, or an empty
  basic_stacktrace object if the initialization of frames_ failed.
*/
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_current_with_skip_depth() {
  // current stack is: [this function, main, (possibly something else, e.g. `_start` from libc)]
  // so it's probably 3 functions deep -- but certainly at least 2 deep.
  auto st = std::stacktrace::current();
  assert(st.size() >= 2);
  auto it     = st.begin();
  auto entry1 = *(it++); // represents this function
  auto entry2 = *(it++); // represents our caller, `main`

  // get current trace again, but skip the 1st
  st = std::stacktrace::current(1, 1);
  assert(st.size() >= 1);
  assert(*st.begin() == entry2);
}

/*
  [4]
  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);
  Postconditions: empty() is true.
*/
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_default_construct() {
  std::stacktrace st;
  assert(st.empty());
}

/*
  [5]
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;
  Effects: alloc is passed to the frames_ constructor.
  Postconditions: empty() is true.
*/
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_construct_with_allocator() {
  test_alloc<std::stacktrace_entry> alloc;
  std::basic_stacktrace<decltype(alloc)> st(alloc);
  assert(st.empty());

  st = std::basic_stacktrace<decltype(alloc)>::current(alloc);
  assert(!st.empty());
}

/*
  [6] basic_stacktrace(const basic_stacktrace& other);
  [7] basic_stacktrace(basic_stacktrace&& other) noexcept;
  [8] basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);
  [9] basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);
  [10] basic_stacktrace& operator=(const basic_stacktrace& other);
  [11] basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
    allocator_traits<Allocator>::is_always_equal::value);
  
  Remarks: Implementations may strengthen the exception specification for these functions
  ([res.on.exception.handling]) by ensuring that empty() is true on failed allocation.
*/
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_copy_move_ctors() {
  using A = std::allocator<std::stacktrace_entry>;
  A alloc;
  auto st = std::basic_stacktrace<A>::current(alloc);

  auto copy_constr = std::basic_stacktrace<A>(st);
  assert(st == copy_constr);

  std::basic_stacktrace<A> copy_assign;
  copy_assign = std::basic_stacktrace<A>(st);
  assert(st == copy_assign);

  auto st2 = test2(alloc);
  assert(st2.size());
  std::basic_stacktrace<A> move_constr(std::move(st2));
  assert(move_constr.size());
  assert(!st2.size());

  auto st3 = test2(alloc);
  assert(st3.size());
  std::basic_stacktrace<A> move_assign;
  move_assign = std::move(st3);
  assert(move_assign.size());
  assert(!st3.size());

  // TODO(stacktrace23): should we add test cases with `select_on_container_copy_construction`?
}

/** Ensure all allocations take place through a given allocator, and that none
 * are sneaking around it by accidentally using malloc or operator new. */
void test_no_malloc_or_new_ex_allocator() {
  // A generic allocator we'll use for everything
  using A = test_alloc<std::stacktrace_entry>;
  A alloc;
  // After this point all stacktrace operations must properly use `alloc`
  malloc_disabled = true;
  // Try all the stack trace operations
  auto st = std::basic_stacktrace<A>(alloc);
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  test_current();
  test_current_with_skip();
  test_current_with_skip_depth();
  test_default_construct();
  test_construct_with_allocator();
  test_copy_move_ctors();
  test_no_malloc_or_new_ex_allocator();

  return 0;
}
