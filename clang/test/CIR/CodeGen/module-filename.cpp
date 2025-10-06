// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// Normally, we try to avoid checking the filename of a test, but that's the
// entire point of this test, so we use a wildcard for the path but check the
// filename.
// CIR: module @"{{.*}}module-filename.cpp"

int main() {
  return 0;
}
