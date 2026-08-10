// RUN: %clang_cc1 -fsyntax-only -verify  %s

#define NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))

#if !__has_attribute(no_sanitize_memory)
#error "Should support no_sanitize_memory"
#endif

void no_analyze() NO_SANITIZE_MEMORY;

void no_analyze_alt() __attribute__((__no_sanitize_memory__));

void no_analyze_args() __attribute__((no_sanitize_memory(1))); // \
  // expected-error {{'no_sanitize_memory' attribute takes no arguments}}

int no_analyze_testfn(int y) NO_SANITIZE_MEMORY;

int no_analyze_testfn(int y) {
  int x NO_SANITIZE_MEMORY = y; // \
    // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
  return x;
}

int no_analyze_test_var NO_SANITIZE_MEMORY; // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}

class NoAnalyzeFoo {
 private:
  int test_field NO_SANITIZE_MEMORY; // \
    // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
  void test_method() NO_SANITIZE_MEMORY;
};

class NO_SANITIZE_MEMORY NoAnalyzeTestClass { // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
};

void no_analyze_params(int lvar NO_SANITIZE_MEMORY); // \
  // expected-error {{'no_sanitize_memory' attribute only applies to functions}}
