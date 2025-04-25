// RUN: %clang -fsanitize=cfi-icall -fno-sanitize-trap=cfi-icall -flto -fvisibility=hidden %s -o %t && not --crash %run %t 2>&1 | FileCheck %s

// RUN: %clang -fsanitize=cfi-icall -fno-sanitize-trap=cfi-icall -fsanitize-recover=cfi-icall -flto -fvisibility=hidden %s -o %t &&  %run %t 2>&1 | FileCheck %s

void f() {}

int main() {
  // CHECK: ubsan: cfi-check-fail by 0x
  ((void (*)(int))f)(42);
}
