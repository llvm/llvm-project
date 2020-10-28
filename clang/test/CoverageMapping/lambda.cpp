// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -x c++ -std=c++11 -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s -main-file-name lambda.cpp | FileCheck %s

// CHECK-LABEL: _Z3fooi:
void foo(int i) { // CHECK: File 0, [[@LINE]]:17 -> {{[0-9]+}}:2 = #0
  auto f = [](int x) {
    return x + 1;
  };

  f(i);
  // Make sure the zero region after the return doesn't escape the lambda.
  // CHECK-NOT: File 0, {{[0-9:]+}} -> [[@LINE+1]]:2 = 0
}

int main(int argc, const char *argv[]) {
  foo(1);
  return 0;
}
