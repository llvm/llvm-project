// RUN: clang-check %s -- -Wdoes-not-exist --serialize-diagnostics %t.dia 2>&1 | FileCheck %s
// RUN: ls %t.dia

// CHECK: warning: unknown warning option '-Wdoes-not-exist'

int main() {
  return 0;
}
