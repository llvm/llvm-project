// RUN: %clang_cc1 -fsyntax-only -verify -std=c23 %s

#if !__has_attribute(zeroize_on_return)
#error "zeroize_on_return is not available via __has_attribute"
#endif

#if !__has_c_attribute(clang::zeroize_on_return)
#error "clang::zeroize_on_return is not available via __has_c_attribute"
#endif

// Both spellings apply to a definition and to a prototype.
[[clang::zeroize_on_return]] void std_definition(void) {}
[[clang::zeroize_on_return]] void std_prototype(void);

__attribute__((zeroize_on_return)) void gnu_definition(void) {}
void gnu_prototype(void) __attribute__((zeroize_on_return));

// The attribute is inheritable, so a prototype carrying it reaches the
// definition in the same translation unit.
[[clang::zeroize_on_return]] void redeclared(void);
void redeclared(void) {}

// It is a single request, so it takes no arguments.
__attribute__((zeroize_on_return(2))) void arg_int(void) {}        // expected-error {{'zeroize_on_return' attribute takes no arguments}}
__attribute__((zeroize_on_return("all"))) void arg_string(void) {} // expected-error {{'zeroize_on_return' attribute takes no arguments}}

// Functions only, and misuse is an error rather than an ignored attribute.
__attribute__((zeroize_on_return)) int global_var;      // expected-error {{'zeroize_on_return' attribute only applies to functions}}
[[clang::zeroize_on_return]] int std_global_var;        // expected-error {{'clang::zeroize_on_return' attribute only applies to functions}}
struct __attribute__((zeroize_on_return)) S { int x; }; // expected-error {{'zeroize_on_return' attribute only applies to functions}}
typedef int my_int __attribute__((zeroize_on_return));  // expected-error {{'zeroize_on_return' attribute only applies to functions}}

// The attribute appertains to the declared variable, not to the function type
// it points at.
__attribute__((zeroize_on_return)) void (*fp)(void); // expected-error {{'zeroize_on_return' attribute only applies to functions}}

void local(void) {
  __attribute__((zeroize_on_return)) int x; // expected-error {{'zeroize_on_return' attribute only applies to functions}}
  (void)x;
}
