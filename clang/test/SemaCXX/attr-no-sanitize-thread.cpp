// RUN: %clang_cc1 -fsyntax-only -verify  %s

#define NO_SANITIZE_THREAD __attribute__((no_sanitize_thread))

#if !__has_attribute(no_sanitize_thread)
#error "Should support no_sanitize_thread"
#endif

void no_analyze_fun() NO_SANITIZE_THREAD;

void no_analyze_alt() __attribute__((__no_sanitize_thread__));

void no_analyze_args() __attribute__((no_sanitize_thread(1))); // \
  // expected-error {{'no_sanitize_thread' attribute takes no arguments}}

int no_analyze_testfn(int y) NO_SANITIZE_THREAD;

int no_analyze_testfn(int y) {
  int x NO_SANITIZE_THREAD = y; // \
    // expected-error {{'no_sanitize_thread' attribute only applies to functions}}
  return x;
}

int no_analyze_test_var NO_SANITIZE_THREAD; // \
  // expected-error {{'no_sanitize_thread' attribute only applies to functions}}

class NoAnalyzeFoo {
 private:
  int test_field NO_SANITIZE_THREAD; // \
    // expected-error {{'no_sanitize_thread' attribute only applies to functions}}
  void test_method() NO_SANITIZE_THREAD;
};

class NO_SANITIZE_THREAD NoAnalyzeTestClass { // \
  // expected-error {{'no_sanitize_thread' attribute only applies to functions}}
};

void no_analyze_params(int lvar NO_SANITIZE_THREAD); // \
  // expected-error {{'no_sanitize_thread' attribute only applies to functions}}
