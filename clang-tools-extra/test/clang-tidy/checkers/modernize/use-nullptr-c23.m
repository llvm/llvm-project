// RUN: %check_clang_tidy %s modernize-use-nullptr %t -- -- -std=c23

#define NULL 0

void test_assignment() {
  int *p1 = NULL;
}
