// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - -O1 -relaxed-aliasing -fsanitize=thread -disable-llvm-optzns %s | \
// RUN:     FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -new-struct-path-tbaa \
// RUN:     -emit-llvm -o - -O1 -relaxed-aliasing -fsanitize=thread -disable-llvm-optzns %s | \
// RUN:     FileCheck %s
//
// Check that we do not create tbaa for instructions generated for copies.

// CHECK-NOT: !tbaa

struct A {
  short s;
  int i;
  char c;
  int j;
};

void copyStruct(A *a1, A *a2) {
  *a1 = *a2;
}

void copyInt(int *a, int *b) {
  *a = *b;
}
