// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

void test_delete_array(int *ptr) {
  delete[] ptr;
}

// LLVM: [[PTR:%[0-9]+]] = load ptr, ptr %{{[0-9]+}}, align 8
// LLVM-NEXT: call void @_ZdaPv(ptr [[PTR]])


int *newmem();
struct cls {
  ~cls();
};
cls::~cls() { delete[] newmem(); }

// LLVM: [[NEWMEM:%[0-9]+]] = call ptr @_Z6newmemv()
// LLVM-NEXT: call void @_ZdaPv(ptr [[NEWMEM]])
