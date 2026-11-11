// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +mte -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify=expected-nomte %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify=expected-nomte %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +mte -x c++ -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -x c++ -std=c++17 -fsyntax-only -verify=expected-nomte %s

#if defined(__aarch64__) && defined(__ARM_FEATURE_MEMORY_TAGGING)
#if !__has_attribute(memtag)
#error "__has_attribute(memtag) should be 1 when MTE is enabled"
#endif
#else
#if __has_attribute(memtag)
#error "__has_attribute(memtag) should be 0 when MTE is disabled"
#endif
#endif

// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) int valid_global;

// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) static int valid_static_global = 1;

// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
extern __attribute__((memtag)) int valid_extern_global;

// Redeclarations inherit the attribute
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
extern __attribute__((memtag)) int redecl_global;
int redecl_global;


// Argument count check
// expected-error@+2 {{'memtag' attribute takes no arguments}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag(1))) int bad_args_global;

// Non-global / TLS subjects
// expected-error@+2 {{'memtag' attribute only applies to non-TLS global variables}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__thread __attribute__((memtag)) int tls_global;

// expected-error@+2 {{'memtag' attribute only applies to non-TLS global variables}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) void func_with_memtag(void) {}

struct S {
  // expected-error@+2 {{'memtag' attribute only applies to non-TLS global variables}}
  // expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
  __attribute__((memtag)) int field;
};

void local_vars(__attribute__((memtag)) int param) { // expected-error {{'memtag' attribute only applies to non-TLS global variables}} \
                                                     // expected-nomte-warning {{unknown attribute 'memtag' ignored}}
  // expected-error@+2 {{'memtag' attribute only applies to non-TLS global variables}}
  // expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
  __attribute__((memtag)) int local_var = 0;

  // Function-local static has global storage, so it is allowed.
  // expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
  __attribute__((memtag)) static int static_local_var = 0;
}

// Constant globals cannot be MTE-tagged
// expected-error@+2 {{'memtag' attribute cannot be applied to a constant global variable}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) const int const_global = 42;

// expected-error@+2 {{'memtag' attribute cannot be applied to a constant global variable}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) const int const_array[4] = {1, 2, 3, 4};

// Pointer to const is fine (the pointer itself is mutable)
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) const char *mutable_ptr_to_const = "hello";

// Const pointer is not fine (the pointer variable itself is const)
// expected-error@+2 {{'memtag' attribute cannot be applied to a constant global variable}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag)) char *const const_ptr = 0;

// Mutual exclusions with section, disable_sanitizer_instrumentation, and no_sanitize("memtag")
// expected-error@+3 {{'section' and 'memtag' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag, section("my_sec"))) int sec_global1;

// expected-error@+3 {{'memtag' and 'section' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((section("my_sec"), memtag)) int sec_global2;

// expected-error@+3 {{'disable_sanitizer_instrumentation' and 'memtag' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag, disable_sanitizer_instrumentation)) int dsi_global1;


// expected-error@+3 {{'no_sanitize' and 'memtag' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag, no_sanitize("memtag"))) int nosan_global1;

// expected-error@+3 {{'memtag' and 'no_sanitize' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((no_sanitize("memtag"), memtag)) int nosan_global2;

// no_sanitize for other sanitizers is allowed alongside memtag
// expected-nomte-warning@+1 {{unknown attribute 'memtag' ignored}}
__attribute__((memtag, no_sanitize("address"))) int nosan_asan_global;

#ifdef __cplusplus
#if defined(__aarch64__) && defined(__ARM_FEATURE_MEMORY_TAGGING)
#if !__has_cpp_attribute(clang::memtag)
#error "__has_cpp_attribute(clang::memtag) should be non-zero when MTE is enabled"
#endif
#else
#if __has_cpp_attribute(clang::memtag)
#error "__has_cpp_attribute(clang::memtag) should be 0 when MTE is disabled"
#endif
#endif

// C++11 [[clang::memtag]] spelling and dependent type check
// expected-nomte-warning@+1 {{unknown attribute 'clang::memtag' ignored}}
[[clang::memtag]] int cxx11_tagged_global;

// Deduced auto const global
// expected-error@+2 {{'clang::memtag' attribute cannot be applied to a constant global variable}}
// expected-nomte-warning@+1 {{unknown attribute 'clang::memtag' ignored}}
[[clang::memtag]] const auto deduced_const_global = 10;

template <typename T>
struct Holder {
  // expected-error@+2 {{'clang::memtag' attribute cannot be applied to a constant global variable}}
  // expected-nomte-warning@+1 {{unknown attribute 'clang::memtag' ignored}}
  [[clang::memtag]] static T value;
};

template <typename T>
T Holder<T>::value = 0;

int use_templates() {
  return Holder<int>::value +
         Holder<const int>::value; // expected-note {{in instantiation of static data member 'Holder<const int>::value' requested here}}
}
#endif
