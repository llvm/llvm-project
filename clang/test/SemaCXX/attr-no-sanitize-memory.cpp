// RUN: %clang_cc1 -fsyntax-only -verify  %s

#define NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))

#if !__has_attribute(no_sanitize_memory)
#error "Should support no_sanitize_memory"
#endif

void noanal_fun() NO_SANITIZE_MEMORY;

void noanal_fun_alt() __attribute__((__no_sanitize_memory__));

void noanal_fun_args() __attribute__((no_sanitize_memory(1))); // \
  // expected-error {{'no_sanitize_memory' attribute takes no arguments}}

int noanal_testfn(int y) NO_SANITIZE_MEMORY;

int noanal_testfn(int y) {
  int x NO_SANITIZE_MEMORY = y; // \
    // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
  return x;
}

int noanal_test_var NO_SANITIZE_MEMORY; // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}

class NoanalFoo {
 private:
  int test_field NO_SANITIZE_MEMORY; // \
    // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
  void test_method() NO_SANITIZE_MEMORY;
};

class NO_SANITIZE_MEMORY NoanalTestClass { // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
};

void noanal_fun_params(int lvar NO_SANITIZE_MEMORY); // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
