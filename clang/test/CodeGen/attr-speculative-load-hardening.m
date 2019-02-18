// RUN: %clang -emit-llvm %s -o - -S | FileCheck %s -check-prefix=SLH

int main() __attribute__((speculative_load_hardening)) {
  return 0;
}

// SLH: @{{.*}}main{{.*}}[[SLH:#[0-9]+]]

// SLH: attributes [[SLH]] = { {{.*}}speculative_load_hardening{{.*}} }
