// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple %itanium_abi_triple -std=c++17 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name default-method.cpp -w %s | FileCheck %s -implicit-check-not="->"

namespace PR39822 {
  struct unique_ptr {
    unique_ptr &operator=(unique_ptr &);
  };

  class foo {
    foo &operator=(foo &);
    unique_ptr convertable_values_[2];
  };

  // CHECK: _ZN7PR398223fooaSERS0_:
  // CHECK-NEXT: File 0, [[@LINE+1]]:28 -> [[@LINE+1]]:29 = #0
  foo &foo::operator=(foo &) = default;
} // namespace PR39822

