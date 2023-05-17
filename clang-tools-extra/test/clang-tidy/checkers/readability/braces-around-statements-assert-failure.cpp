// RUN: not clang-tidy -checks='-*,readability-braces-around-statements' %s --

// Note: this test expects no assert failure happened in clang-tidy.

int test_failure() {
  if (std::rand()) {
  }
}

void test_failure2() {
  for (a b c;;
}
