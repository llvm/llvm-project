// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

// Scalar `operator delete` and array `operator delete[]` use DISTINCT global
// delete wrappers even though their mangled signatures are otherwise identical
// (both YAXPEAX_K@Z): __global_delete for scalar, __global_array_delete for
// array. Conflating them would forward one path to the wrong global operator
// delete. This matches MSVC (validated against cl.exe 19.44), which likewise
// uses __global_delete vs __global_array_delete and shares one empty fallback.

using sz = decltype(sizeof(0));

// Member scalar operator delete -> its deleting destructor's global path uses
// __global_delete.
struct Scalar {
  void *operator new(sz);
  void operator delete(void *, sz);
  virtual ~Scalar();
};
struct ScalarD : Scalar {
  virtual ~ScalarD();
};
Scalar::~Scalar() {}
ScalarD::~ScalarD() {}

// Member array operator delete[] -> its vector deleting destructor's global
// path uses __global_array_delete.
struct Array {
  void *operator new[](sz);
  void operator delete[](void *, sz);
  virtual ~Array();
};
struct ArrayD : Array {
  virtual ~ArrayD();
};
Array::~Array() {}
ArrayD::~ArrayD() {}

// ::delete on class types with non-trivial destructors triggers the forwarding
// bodies for both wrappers.
void test() {
  ::delete new ScalarD();
  ArrayD *a = new ArrayD[2];
  ::delete[] a;
}

// The scalar deleting destructor's global path calls __global_delete.
// CHECK: call void @"?__global_delete@@YAXPEAX_K@Z"(

// It forwards to the scalar ??3@ (::operator delete).
// CHECK: define linkonce_odr void @"?__global_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @"??3@YAXPEAX_K@Z"(ptr %0, i64 %1)

// The vector deleting destructor's global array path calls __global_array_delete.
// CHECK: call void @"?__global_array_delete@@YAXPEAX_K@Z"(

// It forwards to the array ??_V@ (::operator delete[]).
// CHECK: define linkonce_odr void @"?__global_array_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @"??_V@YAXPEAX_K@Z"(ptr %0, i64 %1)

// Both wrappers share a single __empty_global_delete fallback, wired via
// /ALTERNATENAME.
// CHECK-DAG: !{!"/alternatename:?__global_delete@@YAXPEAX_K@Z=?__empty_global_delete@@YAXPEAX_K@Z"}
// CHECK-DAG: !{!"/alternatename:?__global_array_delete@@YAXPEAX_K@Z=?__empty_global_delete@@YAXPEAX_K@Z"}
