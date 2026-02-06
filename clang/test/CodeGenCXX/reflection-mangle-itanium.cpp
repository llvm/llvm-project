// RUN: %clang_cc1 -std=c++26 -freflection -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s -verify

int main() {
  (void)(^^int); // expected-error {{cannot compile this scalar expression yet}}
  return 0;
}
