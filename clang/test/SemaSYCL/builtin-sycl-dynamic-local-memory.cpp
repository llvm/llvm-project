// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -fsyntax-only -verify=sycl -DSYCL %s
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fsyntax-only -verify=sycl -DSYCL %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify=nosycl %s


#ifdef SYCL
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

  // sycl-error@+1 {{cannot initialize a variable of type 'char *' with an rvalue of type '[[clang::sycl_local]] unsigned char *'}}
  char *C = __builtin_sycl_dynamic_local_memory();
}

void args() {
  // sycl-error@+1 {{too many arguments to function call, expected 0, have 1}}
  __builtin_sycl_dynamic_local_memory(1);
}

// The return type is preserved when called in a template.
template <typename T> T [[clang::sycl_local]] *getLocal() {
  static_assert(__is_same(decltype(__builtin_sycl_dynamic_local_memory()),
                          unsigned char [[clang::sycl_local]] *));
  return reinterpret_cast<T [[clang::sycl_local]] *>(
      __builtin_sycl_dynamic_local_memory());
}
void useTemplate() { getLocal<float>(); }
#endif

void notSYCL() {
  // nosycl-error@+1 {{'__builtin_sycl_dynamic_local_memory' is only available in SYCL}}
  __builtin_sycl_dynamic_local_memory();
}
