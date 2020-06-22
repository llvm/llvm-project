// CAUTION: The name of this file should start with `l` for proper tests.
// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s
// REQUIRES: asserts

void test1() {
  // Here __FILE__ macros produces a string with `\` delimiters on Windows
  // and the name of the file starts with `l`.
  char text[] = __FILE__;
}

void test2() {
  // Here `\l` is in the middle of the literal.
  char text[] = "string\\literal";
}

void test() {
  test1();
  test2();
}

// This test is passed if exploded_graph_rewriter handles dot file without errors.
// CHECK: digraph "ExplodedGraph"
// CHECK: clang\\test\\Analysis\\exploded-graph-rewriter\\l_name_starts_with_l.cpp";
// CHECK: char text[] = "string\\literal";
