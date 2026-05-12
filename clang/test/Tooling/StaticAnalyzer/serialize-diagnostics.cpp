// RUN: clang-check %s -- -Wdoes-not-exist --serialize-diagnostics /dev/null 2>&1 | FileCheck %s

// CHECK: warning: unknown warning option '-Wdoes-not-exist'

int main() {
  return 0;
}
