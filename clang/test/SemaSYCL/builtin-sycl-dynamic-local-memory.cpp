// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

// FIXME: Use of the builtin in host code will be diagnosed in a follow-up PR.

// The builtin returns a pointer to unsigned char in local address space.
static_assert(__is_same(decltype(__builtin_sycl_dynamic_local_memory()),
                        unsigned char [[clang::sycl_local]] *));

void conversions() {
  // Implicit conversion to a generic pointer is allowed.
  unsigned char *Generic = __builtin_sycl_dynamic_local_memory();

  // Casting to a pointer to another type in the local address space.
  float [[clang::sycl_local]] *F =
      reinterpret_cast<float [[clang::sycl_local]] *>(
          __builtin_sycl_dynamic_local_memory() + 16);

  // expected-error@+1 {{cannot initialize a variable of type 'char *' with an rvalue of type '[[clang::sycl_local]] unsigned char *'}}
  char *C = __builtin_sycl_dynamic_local_memory();
}

void args() {
  // expected-error@+1 {{too many arguments to function call, expected 0, have 1}}
  __builtin_sycl_dynamic_local_memory(1);
}
