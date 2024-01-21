// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s 2>&1| FileCheck %s

bool func_conditions(bool a, bool b, bool c, bool d, bool e, bool f, bool g) {
  return a && b && c && d && e && f && g;
}

// CHECK: warning: unsupported MC/DC boolean expression; number of conditions{{.*}} exceeds max
