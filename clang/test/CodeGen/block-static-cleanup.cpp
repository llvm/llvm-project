// RUN: %clang_cc1 -std=c++20 -fblocks -emit-llvm -o - %s

struct S {
  ~S();
};

void test() {
  S s1, s2;

  static const int i = ^(void) {
    (void)s1;
    (void)s2;
    return 0;
  }();
}
