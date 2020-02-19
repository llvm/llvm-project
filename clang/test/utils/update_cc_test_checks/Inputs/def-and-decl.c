// Check that the CHECK lines are generated before the definition and not the declaration
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s

int foo();

void empty_function();

int main() {
  empty_function();
  return foo();
}

int foo() {
  return 1;
}

void empty_function() {}
