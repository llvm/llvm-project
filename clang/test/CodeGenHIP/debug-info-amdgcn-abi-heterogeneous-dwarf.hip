// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -O0 -debug-info-kind=limited -gheterogeneous-dwarf -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

// Notes:
// * There is no test involving transparent_union, as this isn't supported in
// C++, and so is not supported in HIP.
// * There is no test involving flexible array members, as this isn't supported
// in C++ without an extension.
// * AMDGCN uses the ItaniumCXXABI, which seems to require one trivial,
// non-deleted copy or move constructor in order to allow Default passing,
// otherwise it selects Indirect. There is a "non-ByVal" form of Indirect,
// which seems to add an extra indirection to avoid a copy, but this is only
// used by the MicrosoftCXXABI, so AFAICT it is impossible to construct for
// AMDGCN.
// * The tests are not exhaustive by any stretch, but try to cover all of the
// relevant corner cases from the perspective of debug info. One notable
// omission is any consideration for return values, as this isn't (currently)
// present in the debug info at all.

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define int8_t char
#define uint8_t unsigned int8_t
#define int16_t short
#define uint16_t unsigned int16_t
#define int32_t int
#define uint32_t unsigned int32_t
#define int64_t long
#define uint64_t unsigned int64_t

// CHECK: %struct.StructEmpty = type { i8 }
struct StructEmpty {};
// CHECK-NEXT: %struct.StructSingleElement = type { i8 }
struct StructSingleElement {
  int8_t Element0;
};
// CHECK-NEXT: %struct.StructSingleElementRecursive = type { %struct.StructSingleElement }
struct StructSingleElementRecursive {
  StructSingleElement Element0;
};
// CHECK-NEXT: %struct.StructTrivialCopyTrivialMove = type { i8 }
struct StructTrivialCopyTrivialMove {
  int8_t Element0;
  __device__ StructTrivialCopyTrivialMove(const StructTrivialCopyTrivialMove &) = default;
  __device__ StructTrivialCopyTrivialMove(StructTrivialCopyTrivialMove &&) = default;
};
// CHECK-NEXT: %struct.StructNoCopyTrivialMove = type { i8 }
struct StructNoCopyTrivialMove {
  int8_t Element0;
  __device__ StructNoCopyTrivialMove(const StructNoCopyTrivialMove &) = delete;
  __device__ StructNoCopyTrivialMove(StructNoCopyTrivialMove &&) = default;
};
// CHECK-NEXT: %struct.StructTrivialCopyNoMove = type { i8 }
struct StructTrivialCopyNoMove {
  int8_t Element0;
  __device__ StructTrivialCopyNoMove(const StructTrivialCopyNoMove &) = default;
  __device__ StructTrivialCopyNoMove(StructTrivialCopyNoMove &&) = delete;
};
// CHECK-NEXT: %struct.StructNoCopyNoMove = type { i8 }
struct StructNoCopyNoMove {
  int8_t Element0;
  __device__ StructNoCopyNoMove(const StructNoCopyNoMove &) = delete;
  __device__ StructNoCopyNoMove(StructNoCopyNoMove &&) = delete;
};
// CHECK-NEXT: %struct.StructNBytes = type { i8, [1 x i8] }
// CHECK-NEXT: %struct.StructNBytes.0 = type { i8, [2 x i8] }
// CHECK-NEXT: %struct.StructNBytes.1 = type { i8, [3 x i8] }
// CHECK-NEXT: %struct.StructNBytes.2 = type { i8, [4 x i8] }
// CHECK-NEXT: %struct.StructNBytes.3 = type { i8, [5 x i8] }
// CHECK-NEXT: %struct.StructNBytes.4 = type { i8, [6 x i8] }
// CHECK-NEXT: %struct.StructNBytes.5 = type { i8, [7 x i8] }
// CHECK-NEXT: %struct.StructNBytes.6 = type { i8, [8 x i8] }
// CHECK-NEXT: %struct.StructNBytes.7 = type { i8, [63 x i8] }
template <unsigned N>
struct StructNBytes {
  static_assert(N > 1, "");
  int8_t Element0;
  int8_t Elements[N - 1u];
};
enum EnumInt8T : int8_t {};
enum EnumUInt8T : uint8_t {};
enum EnumInt16T : int16_t {};
enum EnumUInt16T : uint16_t {};
enum EnumInt32T : int32_t {};
enum EnumUInt32T : uint32_t {};
enum EnumInt64T : int64_t {};
enum EnumUInt64T : uint64_t {};
// CHECK-NEXT: %struct.StructSinglePointerElement = type { ptr }
struct StructSinglePointerElement {
  int32_t *Element0;
};
// CHECK-NEXT: %struct.StructPointerElements = type { ptr, ptr }
struct StructPointerElements {
  int32_t *Element0;
  float *Element1;
};
// CHECK-NEXT: %struct.StructMultipleElements = type { i32, i64 }
struct StructMultipleElements {
  int32_t Element0;
  int64_t Element1;
};

// CHECK: define dso_local void @_Z21Test_Func_StructEmpty11StructEmpty()
// CHECK: %tmp = alloca %struct.StructEmpty, align 1, addrspace(5)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %tmp),
__device__ void Test_Func_StructEmpty(StructEmpty) {}
// CHECK: define dso_local amdgpu_kernel void @_Z21Test_Kern_StructEmpty11StructEmpty(ptr addrspace(4) noundef byref(%struct.StructEmpty) align 1 %0)
// CHECK: %coerce = alloca %struct.StructEmpty, align 1, addrspace(5)
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 1, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_StructEmpty(StructEmpty) {}
// CHECK: define dso_local void @_Z29Test_Func_StructSingleElement19StructSingleElement(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSingleElement, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructSingleElement(StructSingleElement) {}
// CHECK: define dso_local amdgpu_kernel void @_Z29Test_Kern_StructSingleElement19StructSingleElement(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSingleElement, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructSingleElement(StructSingleElement) {}
// CHECK: define dso_local void @_Z38Test_Func_StructSingleElementRecursive28StructSingleElementRecursive(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSingleElementRecursive, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructSingleElementRecursive(StructSingleElementRecursive) {}
// CHECK: define dso_local amdgpu_kernel void @_Z38Test_Kern_StructSingleElementRecursive28StructSingleElementRecursive(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSingleElementRecursive, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructSingleElementRecursive(StructSingleElementRecursive) {}
// CHECK: define dso_local void @_Z38Test_Func_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructTrivialCopyTrivialMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructTrivialCopyTrivialMove(StructTrivialCopyTrivialMove) {}
// CHECK: define dso_local amdgpu_kernel void @_Z38Test_Kern_StructTrivialCopyTrivialMove28StructTrivialCopyTrivialMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructTrivialCopyTrivialMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructTrivialCopyTrivialMove(StructTrivialCopyTrivialMove) {}
// CHECK: define dso_local void @_Z33Test_Func_StructNoCopyTrivialMove23StructNoCopyTrivialMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNoCopyTrivialMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructNoCopyTrivialMove(StructNoCopyTrivialMove) {}
// CHECK: define dso_local amdgpu_kernel void @_Z33Test_Kern_StructNoCopyTrivialMove23StructNoCopyTrivialMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNoCopyTrivialMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructNoCopyTrivialMove(StructNoCopyTrivialMove) {}
// CHECK: define dso_local void @_Z33Test_Func_StructTrivialCopyNoMove23StructTrivialCopyNoMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructTrivialCopyNoMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructTrivialCopyNoMove(StructTrivialCopyNoMove) {}
// CHECK: define dso_local amdgpu_kernel void @_Z33Test_Kern_StructTrivialCopyNoMove23StructTrivialCopyNoMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructTrivialCopyNoMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructTrivialCopyNoMove(StructTrivialCopyNoMove) {}
// CHECK: define dso_local void @_Z28Test_Func_StructNoCopyNoMove18StructNoCopyNoMove(ptr addrspace(5) noundef %[[#ARG:]])
// CHECK: %[[INDIRECT_ADDR:.+]] = alloca ptr, align 8, addrspace(5)
// CHECK: %[[INDIRECT_ADDR_ASCAST:.+]] = addrspacecast ptr addrspace(5) %[[INDIRECT_ADDR]] to ptr
// CHECK: store ptr addrspace(5) %[[#ARG]], ptr %[[INDIRECT_ADDR_ASCAST]], align 8
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[INDIRECT_ADDR]]),
__device__ void Test_Func_StructNoCopyNoMove(StructNoCopyNoMove) {}
// CHECK: define dso_local amdgpu_kernel void @_Z28Test_Kern_StructNoCopyNoMove18StructNoCopyNoMove(i8 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNoCopyNoMove, align 1, addrspace(5)
// CHECK: store i8 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructNoCopyNoMove(StructNoCopyNoMove) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct2Bytes12StructNBytesILj2EE(i16 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes, align 1, addrspace(5)
// CHECK: store i16 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct2Bytes(StructNBytes<2>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct2Bytes12StructNBytesILj2EE(ptr addrspace(4) noundef byref(%struct.StructNBytes) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 2, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct2Bytes(StructNBytes<2>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct3Bytes12StructNBytesILj3EE(i32 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.0, align 1, addrspace(5)
// CHECK: store i32 %.coerce,
// FIXME: I'm unsure why there is a second alloca, and both a store and a memcpy,
// but I assume it relates to padding; ; it should have the same result as e.g.
// the code from Test_Func_Struct2Bytes
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.+}}, ptr align 4 %{{.+}}, i64 3, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct3Bytes(StructNBytes<3>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct3Bytes12StructNBytesILj3EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.0) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.0, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 3, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct3Bytes(StructNBytes<3>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct4Bytes12StructNBytesILj4EE(i32 %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.1, align 1, addrspace(5)
// CHECK: store i32 %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct4Bytes(StructNBytes<4>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct4Bytes12StructNBytesILj4EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.1) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.1, align 1, addrspace(5)
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 4, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct4Bytes(StructNBytes<4>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct5Bytes12StructNBytesILj5EE([2 x i32] %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.2, align 1, addrspace(5)
// CHECK: store [2 x i32] %.coerce,
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.+}}, ptr align 4 %{{.+}}, i64 5, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct5Bytes(StructNBytes<5>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct5Bytes12StructNBytesILj5EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.2) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.2, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 5, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct5Bytes(StructNBytes<5>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct6Bytes12StructNBytesILj6EE([2 x i32] %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.3, align 1, addrspace(5)
// CHECK: store [2 x i32] %.coerce,
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.+}}, ptr align 4 %{{.+}}, i64 6, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct6Bytes(StructNBytes<6>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct6Bytes12StructNBytesILj6EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.3) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.3, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 6, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct6Bytes(StructNBytes<6>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct7Bytes12StructNBytesILj7EE([2 x i32] %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.4, align 1, addrspace(5)
// CHECK: store [2 x i32] %.coerce,
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %{{.+}}, ptr align 4 %{{.+}}, i64 7, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct7Bytes(StructNBytes<7>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct7Bytes12StructNBytesILj7EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.4) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.4, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 7, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct7Bytes(StructNBytes<7>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct8Bytes12StructNBytesILj8EE([2 x i32] %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.5, align 1, addrspace(5)
// CHECK: store [2 x i32] %.coerce,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct8Bytes(StructNBytes<8>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct8Bytes12StructNBytesILj8EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.5) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.5, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 8, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct8Bytes(StructNBytes<8>) {}
// CHECK: define dso_local void @_Z22Test_Func_Struct9Bytes12StructNBytesILj9EE(i8 %.coerce0, [8 x i8] %.coerce1)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructNBytes.6, align 1, addrspace(5)
// CHECK: store i8 %.coerce0,
// CHECK: store [8 x i8] %.coerce1,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_Struct9Bytes(StructNBytes<9>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z22Test_Kern_Struct9Bytes12StructNBytesILj9EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.6) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.6, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 9, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct9Bytes(StructNBytes<9>) {}
// CHECK: define dso_local void @_Z23Test_Func_Struct64Bytes12StructNBytesILj64EE(ptr addrspace(5) noundef byref(%struct.StructNBytes.7) align 1 %0)
__device__ void Test_Func_Struct64Bytes(StructNBytes<64>) {}
// CHECK: define dso_local amdgpu_kernel void @_Z23Test_Kern_Struct64Bytes12StructNBytesILj64EE(ptr addrspace(4) noundef byref(%struct.StructNBytes.7) align 1 %0)
// CHECK: %coerce = alloca %struct.StructNBytes.7, align 1, addrspace(5)
// CHECK-NOT: store
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 1 %{{.+}}, ptr addrspace(4) align 1 %{{.+}}, i64 64, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_Struct64Bytes(StructNBytes<64>) {}
// CHECK: define dso_local void @_Z15Test_Func_Int8Tc(i8 noundef signext %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Int8T(int8_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z15Test_Kern_Int8Tc(i8 noundef %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Int8T(int8_t) {}
// CHECK: define dso_local void @_Z16Test_Func_UInt8Th(i8 noundef zeroext %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_UInt8T(uint8_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z16Test_Kern_UInt8Th(i8 noundef %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_UInt8T(uint8_t) {}
// CHECK: define dso_local void @_Z16Test_Func_Int16Ts(i16 noundef signext %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Int16T(int16_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z16Test_Kern_Int16Ts(i16 noundef %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Int16T(int16_t) {}
// CHECK: define dso_local void @_Z17Test_Func_UInt16Tt(i16 noundef zeroext %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_UInt16T(uint16_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt16Tt(i16 noundef %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_UInt16T(uint16_t) {}
// CHECK: define dso_local void @_Z16Test_Func_Int32Ti(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Int32T(int32_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z16Test_Kern_Int32Ti(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Int32T(int32_t) {}
// CHECK: define dso_local void @_Z17Test_Func_UInt32Tj(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_UInt32T(uint32_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt32Tj(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_UInt32T(uint32_t) {}
// CHECK: define dso_local void @_Z16Test_Func_Int64Tl(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Int64T(int64_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z16Test_Kern_Int64Tl(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Int64T(int64_t) {}
// CHECK: define dso_local void @_Z17Test_Func_UInt64Tm(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_UInt64T(uint64_t) {}
// CHECK: define dso_local amdgpu_kernel void @_Z17Test_Kern_UInt64Tm(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_UInt64T(uint64_t) {}
// CHECK: define dso_local void @_Z19Test_Func_EnumInt8T9EnumInt8T(i8 noundef signext %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumInt8T(EnumInt8T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z19Test_Kern_EnumInt8T9EnumInt8T(i8 noundef %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumInt8T(EnumInt8T) {}
// CHECK: define dso_local void @_Z20Test_Func_EnumUInt8T10EnumUInt8T(i8 noundef zeroext %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumUInt8T(EnumUInt8T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumUInt8T10EnumUInt8T(i8 noundef %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: store i8 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumUInt8T(EnumUInt8T) {}
// CHECK: define dso_local void @_Z20Test_Func_EnumInt16T10EnumInt16T(i16 noundef signext %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumInt16T(EnumInt16T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt16T10EnumInt16T(i16 noundef %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumInt16T(EnumInt16T) {}
// CHECK: define dso_local void @_Z21Test_Func_EnumUInt16T11EnumUInt16T(i16 noundef zeroext %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumUInt16T(EnumUInt16T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt16T11EnumUInt16T(i16 noundef %0)
// CHECK: %.addr = alloca i16, align 2, addrspace(5)
// CHECK: store i16 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumUInt16T(EnumUInt16T) {}
// CHECK: define dso_local void @_Z20Test_Func_EnumInt32T10EnumInt32T(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumInt32T(EnumInt32T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt32T10EnumInt32T(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumInt32T(EnumInt32T) {}
// CHECK: define dso_local void @_Z21Test_Func_EnumUInt32T11EnumUInt32T(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumUInt32T(EnumUInt32T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt32T11EnumUInt32T(i32 noundef %0)
// CHECK: %.addr = alloca i32, align 4, addrspace(5)
// CHECK: store i32 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumUInt32T(EnumUInt32T) {}
// CHECK: define dso_local void @_Z20Test_Func_EnumInt64T10EnumInt64T(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumInt64T(EnumInt64T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z20Test_Kern_EnumInt64T10EnumInt64T(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumInt64T(EnumInt64T) {}
// CHECK: define dso_local void @_Z21Test_Func_EnumUInt64T11EnumUInt64T(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_EnumUInt64T(EnumUInt64T) {}
// CHECK: define dso_local amdgpu_kernel void @_Z21Test_Kern_EnumUInt64T11EnumUInt64T(i64 noundef %0)
// CHECK: %.addr = alloca i64, align 8, addrspace(5)
// CHECK: store i64 %0,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_EnumUInt64T(EnumUInt64T) {}
// CHECK: define dso_local void @_Z27Test_Func_PromotableIntegerb(i1 noundef zeroext %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: %frombool = zext i1 %0 to i8
// CHECK: store i8 %frombool,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_PromotableInteger(bool) {}
// CHECK: define dso_local amdgpu_kernel void @_Z27Test_Kern_PromotableIntegerb(i1 noundef %0)
// CHECK: %.addr = alloca i8, align 1, addrspace(5)
// CHECK: %frombool = zext i1 %0 to i8
// CHECK: store i8 %frombool,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_PromotableInteger(bool) {}
// CHECK: define dso_local void @_Z17Test_Func_PointerPi(ptr noundef %0)
// CHECK: %.addr = alloca ptr, align 8, addrspace(5)
// CHECK: store ptr %0, ptr %.addr.ascast, align 8
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Pointer(int32_t *) {}
// CHECK: define dso_local amdgpu_kernel void @_Z17Test_Kern_PointerPi(ptr addrspace(1) noundef %.coerce)
// CHECK: %.addr = alloca ptr, align 8, addrspace(5)
// FIXME: There is a store, load, store sequence through another alloca here,
// which I don't understand the intent of
// CHECK: store ptr
// call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Pointer(int32_t *) {}
// CHECK: define dso_local void @_Z19Test_Func_ReferenceRi(ptr noundef nonnull align 4 dereferenceable(4) %0)
// CHECK: %.addr = alloca ptr, align 8, addrspace(5)
// CHECK: store ptr %0, ptr %.addr.ascast, align 8
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__device__ void Test_Func_Reference(int32_t &) {}
// CHECK: define dso_local amdgpu_kernel void @_Z19Test_Kern_ReferenceRi(ptr addrspace(1) noundef nonnull align 4 dereferenceable(4) %.coerce)
// CHECK: %.addr = alloca ptr, align 8, addrspace(5)
// FIXME: There is a store, load, store sequence through another alloca here,
// which I don't understand the intent of
// CHECK: store ptr
// call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %.addr),
__global__ void Test_Kern_Reference(int32_t &) {}
// CHECK: define dso_local void @_Z36Test_Func_StructSinglePointerElement26StructSinglePointerElement(ptr %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSinglePointerElement, align 8, addrspace(5)
// CHECK: store ptr %.coerce,
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructSinglePointerElement(StructSinglePointerElement) {}
// CHECK: define dso_local amdgpu_kernel void @_Z36Test_Kern_StructSinglePointerElement26StructSinglePointerElement(ptr addrspace(1) %.coerce)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructSinglePointerElement, align 8, addrspace(5)
// CHECK: store ptr
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__global__ void Test_Kern_StructSinglePointerElement(StructSinglePointerElement) {}
// CHECK: define dso_local void @_Z31Test_Func_StructPointerElements21StructPointerElements(ptr %.coerce0, ptr %.coerce1)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructPointerElements, align 8, addrspace(5)
// CHECK: store ptr
// CHECK: store ptr
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_StructPointerElements(StructPointerElements) {}
// CHECK: define dso_local amdgpu_kernel void @_Z31Test_Kern_StructPointerElements21StructPointerElements(ptr addrspace(4) noundef byref(%struct.StructPointerElements) align 8 %{{.+}})
// CHECK: %coerce = alloca %struct.StructPointerElements, align 8, addrspace(5)
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 8 %{{.+}}, ptr addrspace(4) align 8 %{{.+}}, i64 16, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_StructPointerElements(StructPointerElements) {}
// CHECK: define dso_local void @_Z37Test_Func_ParamRegLimitExpandedStructlllllli22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i32 noundef %6, i32 %.coerce0, i64 %.coerce1)
// CHECK: %[[#ALLOCA:]] = alloca %struct.StructMultipleElements, align 8, addrspace(5)
// CHECK: store i32 %.coerce0,
// CHECK: store i64 %.coerce1,
// CHECK-NOT: {{.*}}memcpy{{.*}}
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %[[#ALLOCA]]),
__device__ void Test_Func_ParamRegLimitExpandedStruct(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, StructMultipleElements) {}
// CHECK: define dso_local amdgpu_kernel void @_Z37Test_Kern_ParamRegLimitExpandedStructlllllli22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i32 noundef %6, ptr addrspace(4) noundef byref(%struct.StructMultipleElements) align 8 %7)
// CHECK: %coerce = alloca %struct.StructMultipleElements, align 8, addrspace(5)
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 8 %{{.+}}, ptr addrspace(4) align 8 %{{.+}}, i64 16, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_ParamRegLimitExpandedStruct(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, StructMultipleElements) {}
// CHECK: define dso_local void @_Z39Test_Func_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, ptr addrspace(5) noundef byref(%struct.StructMultipleElements) align 8 %7)
__device__ void Test_Func_ParamRegLimitUnexpandedStruct(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, StructMultipleElements) {}
// CHECK: define dso_local amdgpu_kernel void @_Z39Test_Kern_ParamRegLimitUnexpandedStructlllllll22StructMultipleElements(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5, i64 noundef %6, ptr addrspace(4) noundef byref(%struct.StructMultipleElements) align 8 %7)
// CHECK: %coerce = alloca %struct.StructMultipleElements, align 8, addrspace(5)
// CHECK: call void @llvm.memcpy.p0.p4.i64(ptr align 8 %{{.+}}, ptr addrspace(4) align 8 %{{.+}}, i64 16, i1 false)
// CHECK: call void @llvm.dbg.def(metadata !{{[0-9]+}}, metadata ptr addrspace(5) %coerce),
__global__ void Test_Kern_ParamRegLimitUnexpandedStruct(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, StructMultipleElements) {}
