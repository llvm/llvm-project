// RUN: %check_clang_tidy %s llvmlibc-callee-namespace %t

#define OTHER_MACRO_NAMESPACE custom_namespace
namespace OTHER_MACRO_NAMESPACE {
  void wrong_name_macro_func() {}
}

namespace __llvm_libc {
  void right_name_no_macro_func() {}
}

#define LIBC_NAMESPACE __llvm_libc_xyz
namespace LIBC_NAMESPACE {
namespace nested {
void nested_func() {}
} // namespace nested
void libc_api_func() {}

struct libc_api_struct {
  int operator()() const { return 0; }
};
} // namespace __llvm_libc

// Emulate a function from the public headers like string.h
void libc_api_func() {}

// Emulate a function specifically allowed by the exception list.
void malloc() {}

// Emulate a non-trivially named symbol.
struct global_struct {
  int operator()() const { return 0; }
};

namespace LIBC_NAMESPACE {
void Test() {
  // Allow calls with the fully qualified name.
  LIBC_NAMESPACE::libc_api_func();
  LIBC_NAMESPACE::nested::nested_func();
  void (*qualifiedPtr)(void) = LIBC_NAMESPACE::libc_api_func;
  qualifiedPtr();

  // Should not trigger on compiler provided function calls.
  (void)__builtin_abs(-1);

  // Bare calls are allowed as long as they resolve to the correct namespace.
  libc_api_func();
  nested::nested_func();
  void (*barePtr)(void) = LIBC_NAMESPACE::libc_api_func;
  barePtr();

  // Allow calling entities defined in the namespace.
  LIBC_NAMESPACE::libc_api_struct{}();

  // Disallow calling into global namespace for implemented entrypoints.
  ::libc_api_func();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'libc_api_func' must resolve to a function declared within the namespace defined by the 'LIBC_NAMESPACE' macro
  // CHECK-MESSAGES: :25:6: note: resolves to this declaration

  // Disallow indirect references to functions in global namespace.
  void (*badPtr)(void) = ::libc_api_func;
  badPtr();
  // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: 'libc_api_func' must resolve to a function declared within the namespace defined by the 'LIBC_NAMESPACE' macro
  // CHECK-MESSAGES: :25:6: note: resolves to this declaration

  // Allow calling into global namespace for specific functions.
  ::malloc();

  // Disallow calling on entities that are not in the namespace, but make sure
  // no crashes happen.
  global_struct{}();
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'operator()' must resolve to a function declared within the namespace defined by the 'LIBC_NAMESPACE' macro
  // CHECK-MESSAGES: :32:7: note: resolves to this declaration

  OTHER_MACRO_NAMESPACE::wrong_name_macro_func();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'wrong_name_macro_func' must resolve to a function declared within the namespace defined by the 'LIBC_NAMESPACE' macro
  // CHECK-MESSAGES: :3:31: note: expanded from macro 'OTHER_MACRO_NAMESPACE'
  // CHECK-MESSAGES: :5:8: note: resolves to this declaration
  
  __llvm_libc::right_name_no_macro_func();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'right_name_no_macro_func' must resolve to a function declared within the namespace defined by the 'LIBC_NAMESPACE' macro
  // CHECK-MESSAGES: :9:8: note: resolves to this declaration
  
}

} // namespace __llvm_libc
