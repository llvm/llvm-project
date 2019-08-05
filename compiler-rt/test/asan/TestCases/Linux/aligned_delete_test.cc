// RUN: %clangxx_asan -std=c++1z -faligned-allocation -fsanitize-recover=address -O0 %s -o %t
// RUN: %env_asan_opts=new_delete_type_mismatch=1:halt_on_error=false:detect_leaks=false %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=new_delete_type_mismatch=0                                        %run %t

// RUN: %clangxx_asan -std=c++1z -faligned-allocation -fsized-deallocation -fsanitize-recover=address -O0 %s -o %t
// RUN: %env_asan_opts=new_delete_type_mismatch=1:halt_on_error=false:detect_leaks=false %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=new_delete_type_mismatch=0                                        %run %t

#include <stdio.h>

// Define all new/delete to do not depend on the version provided by the
// platform. The implementation is provided by ASan anyway.

namespace std {
struct nothrow_t {};
static const nothrow_t nothrow;
enum class align_val_t : size_t {};
}  // namespace std

void *operator new(size_t);
void *operator new[](size_t);
void *operator new(size_t, std::nothrow_t const&);
void *operator new[](size_t, std::nothrow_t const&);
void *operator new(size_t, std::align_val_t);
void *operator new[](size_t, std::align_val_t);
void *operator new(size_t, std::align_val_t, std::nothrow_t const&);
void *operator new[](size_t, std::align_val_t, std::nothrow_t const&);

void operator delete(void*) throw();
void operator delete[](void*) throw();
void operator delete(void*, std::nothrow_t const&);
void operator delete[](void*, std::nothrow_t const&);
void operator delete(void*, size_t) throw();
void operator delete[](void*, size_t) throw();
void operator delete(void*, std::align_val_t) throw();
void operator delete[](void*, std::align_val_t) throw();
void operator delete(void*, std::align_val_t, std::nothrow_t const&);
void operator delete[](void*, std::align_val_t, std::nothrow_t const&);
void operator delete(void*, size_t, std::align_val_t) throw();
void operator delete[](void*, size_t, std::align_val_t) throw();


template<typename T>
inline T* break_optimization(T *arg) {
  __asm__ __volatile__("" : : "r" (arg) : "memory");
  return arg;
}


struct S12 { int a, b, c; };
struct alignas(128) S12_128 { int a, b, c; };
struct alignas(256) S12_256 { int a, b, c; };
struct alignas(512) S1024_512 { char a[1024]; };
struct alignas(1024) S1024_1024 { char a[1024]; };


int main(int argc, char **argv) {
  // Check the mismatched calls only, all the valid cases are verified in
  // test/sanitizer_common/TestCases/Linux/new_delete_test.cpp.

  operator delete(break_optimization(new S12_128), std::nothrow);
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   128 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  operator delete(break_optimization(new S12_128), sizeof(S12_128));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   128 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  operator delete[](break_optimization(new S12_128[100]), std::nothrow);
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   128 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  operator delete[](break_optimization(new S12_128[100]), sizeof(S12_128[100]));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   128 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  // Various mismatched alignments.

  delete break_optimization(reinterpret_cast<S12*>(new S12_256));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   256 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  delete break_optimization(reinterpret_cast<S12_256*>(new S12));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   default-aligned;
  // CHECK:  alignment of the deallocated type: 256 bytes.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  delete break_optimization(reinterpret_cast<S12_128*>(new S12_256));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   256 bytes;
  // CHECK:  alignment of the deallocated type: 128 bytes.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  delete [] break_optimization(reinterpret_cast<S12*>(new S12_256[100]));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   256 bytes;
  // CHECK:  alignment of the deallocated type: default-aligned.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  delete [] break_optimization(reinterpret_cast<S12_256*>(new S12[100]));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   default-aligned;
  // CHECK:  alignment of the deallocated type: 256 bytes.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  delete [] break_optimization(reinterpret_cast<S12_128*>(new S12_256[100]));
  // CHECK: AddressSanitizer: new-delete-type-mismatch
  // CHECK:  object passed to delete has wrong type:
  // CHECK:  alignment of the allocated type:   256 bytes;
  // CHECK:  alignment of the deallocated type: 128 bytes.
  // CHECK: SUMMARY: AddressSanitizer: new-delete-type-mismatch

  // Push ASan limits, the current limitation is that it cannot differentiate
  // alignments above 512 bytes.
  fprintf(stderr, "Checking alignments >= 512 bytes\n");
  delete break_optimization(reinterpret_cast<S1024_512*>(new S1024_1024));
  fprintf(stderr, "Done\n");
  // CHECK: Checking alignments >= 512 bytes
  // CHECK-NEXT: Done
}
