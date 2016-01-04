// RUN: %clang_cc1 -fsyntax-only -verify  %s

#define NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))

#if !__has_attribute(no_sanitize_address)
#error "Should support no_sanitize_address"
#endif

void noanal_fun() NO_SANITIZE_ADDRESS;

void noanal_fun_alt() __attribute__((__no_sanitize_address__));

void noanal_fun_args() __attribute__((no_sanitize_address(1))); // \
  // expected-error {{'no_sanitize_address' attribute takes no arguments}}

int noanal_testfn(int y) NO_SANITIZE_ADDRESS;

int noanal_testfn(int y) {
  int x NO_SANITIZE_ADDRESS = y; // \
    // expected-error {{'no_sanitize_address' attribute only applies to functions}}
  return x;
}

int noanal_test_var NO_SANITIZE_ADDRESS; // \
  // expected-error {{'no_sanitize_address' attribute only applies to functions}}

class NoanalFoo {
 private:
  int test_field NO_SANITIZE_ADDRESS; // \
    // expected-error {{'no_sanitize_address' attribute only applies to functions}}
  void test_method() NO_SANITIZE_ADDRESS;
};

class NO_SANITIZE_ADDRESS NoanalTestClass { // \
  // expected-error {{'no_sanitize_address' attribute only applies to functions}}
};

void noanal_fun_params(int lvar NO_SANITIZE_ADDRESS); // \
  // expected-error {{'no_sanitize_address' attribute only applies to functions}}
