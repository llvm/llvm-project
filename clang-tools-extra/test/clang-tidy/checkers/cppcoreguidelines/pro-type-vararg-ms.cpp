// Purpose:
// Ensure that the 'cppcoreguidelines-pro-type-vararg' check works with the
// built-in va_list on Windows systems.

// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-vararg %t -- --extra-arg=--target=x86_64-windows

void test_ms_va_list(int a, ...) {
  __builtin_ms_va_list ap;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare variables of type va_list; use variadic templates instead
  __builtin_ms_va_start(ap, a);
  int b = __builtin_va_arg(ap, int);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not use va_arg to define c-style vararg functions; use variadic templates instead
  __builtin_ms_va_end(ap);
}

void test_typedefs(int a, ...) {
  typedef __builtin_ms_va_list my_va_list1;
  my_va_list1 ap1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare variables of type va_list; use variadic templates instead

  using my_va_list2 = __builtin_ms_va_list;
  my_va_list2 ap2;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare variables of type va_list; use variadic templates instead
}
