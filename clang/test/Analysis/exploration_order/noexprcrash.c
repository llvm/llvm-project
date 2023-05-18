// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config exploration_strategy=unexplored_first %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -analyzer-config exploration_strategy=dfs %s

extern void clang_analyzer_eval(int);

typedef struct { char a; } b;
int c(b* input) {
    int x = (input->a ?: input) ? 1 : 0; // expected-warning{{pointer/integer type mismatch}}
    if (input->a) {
      // FIXME: The value should actually be "TRUE",
      // but is incorrect due to a bug.
      clang_analyzer_eval(x); // expected-warning{{FALSE}}
    } else {
      clang_analyzer_eval(x); // expected-warning{{TRUE}}
    }
    return x;
}
