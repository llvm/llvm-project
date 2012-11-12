// RUN: %clang_cc1 -emit-llvm -O1 < %s | grep "ret i32 0"

int a(void) {
  return ++(struct x {unsigned x : 2;}){3}.x;
}


