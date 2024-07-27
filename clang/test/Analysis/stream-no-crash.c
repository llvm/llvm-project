// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Stream -verify %s

// This test is isolate since it uses line markers to repro the problem.
// The test is expected to find the issues noted below without crashing.

# 1 "" 1
# 1 "" 1
# 1 "" 1
# 1 "" 1 3
typedef FILE;
extern *stdout;
char a;
*fopen();
# 0 "" 2
# 7 "" 2
# 7 "" 2
# 7 "" 2
void b() {
  fopen(&a, "");
  int c = stdout && c;
  b();
}
// expected-warning@-3{{Assigned value is garbage or undefined}}
// expected-warning@-4{{Opened stream never closed. Potential resource leak}}

