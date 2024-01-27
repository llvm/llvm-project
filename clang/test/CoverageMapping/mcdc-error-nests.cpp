// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s

// "Split-nest" -- boolean expressions within boolean expressions.
extern bool bar(bool);
bool func_split_nest(bool a, bool b, bool c, bool d, bool e, bool f, bool g) {
  bool res = a && b && c && bar(d && e) && f && g;
  return bar(res);
}

// CHECK: warning: unsupported MC/DC boolean expression; contains an operation with a nested boolean expression.
