// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

// PR447

namespace nm {
  struct str {
    friend void foo(int arg = 0) {};
  };
}
