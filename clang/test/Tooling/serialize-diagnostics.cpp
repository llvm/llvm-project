// RUN: clang-check %s -- -Wunknown-warning-option --serialize-diagnostics %t.dia 2>&1 | FileCheck %s --allow-empty
// RUN: ls %t.dia

int main() {
  return 0;
}
