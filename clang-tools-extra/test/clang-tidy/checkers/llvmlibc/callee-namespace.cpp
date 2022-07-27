// RUN: %check_clang_tidy %s llvmlibc-callee-namespace %t

namespace __llvm_libc {
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

namespace __llvm_libc {
void Test() {
  // Allow calls with the fully qualified name.
  __llvm_libc::libc_api_func();
  __llvm_libc::nested::nested_func();
  void (*qualifiedPtr)(void) = __llvm_libc::libc_api_func;
  qualifiedPtr();

  // Should not trigger on compiler provided function calls.
  (void)__builtin_abs(-1);

  // Bare calls are allowed as long as they resolve to the correct namespace.
  libc_api_func();
  nested::nested_func();
  void (*barePtr)(void) = __llvm_libc::libc_api_func;
  barePtr();

  // Allow calling entities defined in the namespace.
  __llvm_libc::libc_api_struct{}();

  // Disallow calling into global namespace for implemented entrypoints.
  ::libc_api_func();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'libc_api_func' must resolve to a function declared within the '__llvm_libc' namespace
  // CHECK-MESSAGES: :15:6: note: resolves to this declaration

  // Disallow indirect references to functions in global namespace.
  void (*badPtr)(void) = ::libc_api_func;
  badPtr();
  // CHECK-MESSAGES: :[[@LINE-2]]:26: warning: 'libc_api_func' must resolve to a function declared within the '__llvm_libc' namespace
  // CHECK-MESSAGES: :15:6: note: resolves to this declaration

  // Allow calling into global namespace for specific functions.
  ::malloc();

  // Disallow calling on entities that are not in the namespace, but make sure
  // no crashes happen.
  global_struct{}();
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'operator()' must resolve to a function declared within the '__llvm_libc' namespace
  // CHECK-MESSAGES: :22:7: note: resolves to this declaration
}

} // namespace __llvm_libc
