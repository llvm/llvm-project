// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: main
int main(int argc, char **argv) {
  // CHECK: load i8, ptr %
  // CHECK-NEXT: icmp ne i8 %{{.+}}, 0
  bool b = (bool &)argv[argc][1];
  return b;
}
