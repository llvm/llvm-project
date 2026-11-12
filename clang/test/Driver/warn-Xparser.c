/// Some macOS projects use -Xparser.
// RUN: %clang -c -o /dev/null -Xparser %s 2>&1 | FileCheck %s

// CHECK: warning: argument unused during compilation: '-Xparser' [-Wunused-command-line-argument]

void f(void) {}
