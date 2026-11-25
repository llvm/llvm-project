// RUN: %clang_analyze_cc1 -x c++ -std=c++17 -analyzer-checker=core,deadcode.DeadStores,debug.ExprInspection -verify %s

void clang_analyzer_dump(const char *);

// The cleanup handler runs before the destructor of the same variable.
static const char *dtor_state;

struct WithDtor {
  ~WithDtor() { dtor_state = "dtor"; }
};

static void cleanup_before_dtor(struct WithDtor *) {
  clang_analyzer_dump(dtor_state); // expected-warning {{"cleanup"}}
}

void cleanup_runs_before_destructor(void) {
  struct WithDtor w __attribute__((cleanup(cleanup_before_dtor)));
  dtor_state = "cleanup";
}

// The cleanup handler runs when the lambda call operator is inlined.
static const char *lambda_state;

static void set_lambda_state(int *) {
  lambda_state = "cleanup";
}

void cleanup_in_lambda_body(void) {
  auto lam = []() {
    int z __attribute__((cleanup(set_lambda_state)));
    z = 1;
  };
  lambda_state = "before";
  clang_analyzer_dump(lambda_state); // expected-warning {{"before"}}
  lam();
  clang_analyzer_dump(lambda_state); // expected-warning {{"cleanup"}}
}
