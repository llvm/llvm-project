// Test that branch regions are generated for conditions in function template
// instantiations.

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-templates.cpp %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name branch-templates.cpp %s | FileCheck %s

template<typename T>
void unused(T x) {
  return;
}

template<typename T>
int func(T x) {
  if(x)
    return 0;
  else
    return 1;
  int j = 1;
}

int main() {
  func<int>(0);
  func<bool>(true);
  func<float>(0.0);
  return 0;
}

// CHECK-LABEL: _Z4funcIiEiT_:
// CHECK: Branch,File 0, [[@LINE-15]]:6 -> [[@LINE-15]]:7 = #1, (#0 - #1)
// CHECK-LABEL: _Z4funcIbEiT_:
// CHECK: Branch,File 0, [[@LINE-17]]:6 -> [[@LINE-17]]:7 = #1, (#0 - #1)
// CHECK-LABEL: _Z4funcIfEiT_:
// CHECK: Branch,File 0, [[@LINE-19]]:6 -> [[@LINE-19]]:7 = #1, (#0 - #1)
