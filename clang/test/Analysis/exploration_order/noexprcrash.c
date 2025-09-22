// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify=common,ufirst -analyzer-config exploration_strategy=unexplored_first %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify=common,dfs -analyzer-config exploration_strategy=dfs %s

extern void clang_analyzer_eval(int);

typedef struct { char a; } b;
int c(b* input) {
    int x = (input->a ?: input) ? 1 : 0; // common-warning{{pointer/integer type mismatch}}
    if (input->a) {
      // FIXME: The value should actually be "TRUE",
      // but is incorrect due to a bug.
      // dfs-warning@+1 {{FALSE}} ufirst-warning@+1 {{TRUE}}
      clang_analyzer_eval(x);
    } else {
      clang_analyzer_eval(x); // common-warning{{TRUE}}
    }
    return x;
}
