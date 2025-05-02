// RUN: %clang -fsanitize=cfi-icall -fno-sanitize-trap=cfi-icall                              -fuse-ld=lld -flto -fvisibility=hidden %s -o %t && not --crash %run %t 2>&1 | FileCheck %s
// RUN: %clang -fsanitize=cfi-icall -fno-sanitize-trap=cfi-icall -fsanitize-recover=cfi-icall -fuse-ld=lld -flto -fvisibility=hidden %s -o %t &&             %run %t 2>&1 | FileCheck %s

// REQUIRES: lld-available, cfi

void f() {}

int main() {
  // CHECK: ubsan: cfi-check-fail by 0x
  ((void (*)(int))f)(42);
}
