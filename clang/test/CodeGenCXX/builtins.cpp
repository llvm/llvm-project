// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// PR8839
extern "C" char memmove();

int main() {
  // CHECK: call signext i8 @memmove()
  return memmove();
}
