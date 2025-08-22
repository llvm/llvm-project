// RUN: %clangxx -g -fsanitize=null -fsanitize-trap=all -fsanitize-annotate-debug-info=all -O2 -std=c++17 -c -o /dev/null %s

struct foo {
  foo(int, long, const int & = int());
} foo(0, 0);
