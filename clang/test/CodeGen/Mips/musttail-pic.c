// RUN: %clang_cc1 %s -triple mipsel-unknown-linux-gnu -pic-level 2 \
// RUN:   -fhalf-no-semantic-interposition -o /dev/null -emit-llvm -verify

// Test musttail behavior in PIC mode with semantic interposition.

int external_defined(int i) { return i; }
static int static_func(int i) { return i; }
__attribute__((visibility("hidden"))) int hidden_defined(int i) { return i; }

int call_external(int i) {
  // expected-error@+1 {{'musttail' attribute for this call is impossible because calls outside the current linkage unit cannot be tail called on MIPS}}
  [[clang::musttail]] return external_defined(i);
}

int call_static(int i) {
  [[clang::musttail]] return static_func(i);
}

int call_hidden(int i) {
  [[clang::musttail]] return hidden_defined(i);
}
