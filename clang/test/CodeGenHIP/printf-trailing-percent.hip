// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm -fcuda-is-device \
// RUN:   -Wno-format -mprintf-kind=hostcall -o - %s \
// RUN:   | FileCheck --check-prefix=HOSTCALL %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm -fcuda-is-device \
// RUN:   -Wno-format -mprintf-kind=buffered -o - %s \
// RUN:   | FileCheck --check-prefix=BUFFERED %s

// A format string ending in an incomplete specifier must not be read past its
// end, and the specifiers before it must still be found.

#define __device__ __attribute__((device))

extern "C" __device__ int printf(const char *format, ...);

__device__ int trailing_percent(const char *s) { return printf("%s%", s); }

// The %s argument is printed as a string, not as a pointer.

// HOSTCALL-LABEL: define {{.*}} @_Z16trailing_percentPKc(
// HOSTCALL:         call i64 @__ockl_printf_begin(i64 0)
// HOSTCALL:         call i64 @__ockl_printf_append_string_n(i64 %{{.*}}, ptr {{.*}}@.str{{.*}}, i64 %{{.*}}, i32 0)
// HOSTCALL:         call i64 @__ockl_printf_append_string_n(i64 %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i32 1)
// HOSTCALL-NOT:     call i64 @__ockl_printf_append_args

// BUFFERED-LABEL: define {{.*}} @_Z16trailing_percentPKc(
// BUFFERED:         call ptr addrspace(1) @__printf_alloc(
// BUFFERED:         call void @llvm.memcpy.p1.p0.i64(
// BUFFERED-NOT:     ptrtoint
// BUFFERED:       !{!"0:0:{{[0-9a-f]+}},%s%"}
