// RUN: %clang_cc1 %s -triple mipsel-unknown-linux-gnu -o /dev/null -emit-llvm -verify

// Test that hidden and protected visibility functions can be tail called
// because they are guaranteed to be DSO-local.

extern int hidden_func(int i) __attribute__((visibility("hidden")));
extern int protected_func(int i) __attribute__((visibility("protected")));
extern int default_func(int i);

int call_hidden(int i) {
  // expected-no-diagnostics
  [[clang::musttail]] return hidden_func(i);
}

int call_protected(int i) {
  [[clang::musttail]] return protected_func(i);
}
