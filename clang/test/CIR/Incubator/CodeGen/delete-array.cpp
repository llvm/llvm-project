// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void test_delete_array(int *ptr) {
  delete[] ptr;
}

// CHECK: cir.delete.array 
