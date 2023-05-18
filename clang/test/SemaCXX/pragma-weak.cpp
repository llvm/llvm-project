// RUN: %clang_cc1 %s

#pragma weak foo
static void foo();
extern "C" {
  void foo() {
  };
}

extern "C" int Test;
#pragma weak test = Test
