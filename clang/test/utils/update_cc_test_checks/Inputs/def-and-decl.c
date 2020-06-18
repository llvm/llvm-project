// Check that the CHECK lines are generated before the definition and not the declaration
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s

int foo(int arg);

void empty_function(void);

int main() {
  empty_function();
  return foo(1);
}

int foo(int arg) {
  return arg;
}

void empty_function(void) {}
