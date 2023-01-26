// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck --enable-var-scope %s

#define __device__ __attribute__((device))

extern "C" __device__ int printf(const char *format, ...);

__device__ int foo1() {
  const char *s = "hello world";
  return printf("%.*f %*.*s %p\n", 8, 3.14159, 8, 4, s, s);
}

// CHECK-LABEL: @_Z4foo1v()
// CHECK: [[BEGIN:%.*]]   = call i64 @__ockl_printf_begin(i64 0)
// CHECK: [[STRLEN1:%.*]] = phi i64 [ %{{[^,]*}}, %{{[^ ]*}} ], [ 0, %{{[^ ]*}} ]
// CHECK: [[APPEND1:%.*]] = call i64 @__ockl_printf_append_string_n(i64 [[BEGIN]], {{.*}}, i64 [[STRLEN1]], i32 0)
// CHECK: [[APPEND2:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND1]], i32 1, i64 8, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// CHECK: [[APPEND3:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND2]], i32 1, i64 4614256650576692846, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// CHECK: [[APPEND4:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND3]], i32 1, i64 8, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// CHECK: [[APPEND5:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND4]], i32 1, i64 4, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// CHECK: [[STRLEN2:%.*]] = phi i64 [ %{{[^,]*}}, %{{[^ ]*}} ], [ 0, %{{[^ ]*}} ]
// CHECK: [[APPEND6:%.*]] = call i64 @__ockl_printf_append_string_n(i64 [[APPEND5]], {{.*}}, i64 [[STRLEN2]], i32 0)
// CHECK: [[PTR2INT:%.*]] = ptrtoint ptr %{{.*}} to i64
// CHECK: [[APPEND7:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND6]], i32 1, i64 [[PTR2INT]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// CHECK: [[RETURN:%.*]]  = trunc i64 [[APPEND7]] to i32
// CHECK: ret i32 [[RETURN]]

__device__ char *dstr;

__device__ int foo2() {
  return printf("%s %p\n", dstr, dstr);
}

// CHECK-LABEL: @_Z4foo2v()
// CHECK: [[BEGIN:%.*]]   = call i64 @__ockl_printf_begin(i64 0)
// CHECK: [[STRLEN1:%.*]] = phi i64 [ %{{[^,]*}}, %{{[^ ]*}} ], [ 0, %{{[^ ]*}} ]
// CHECK: [[APPEND1:%.*]] = call i64 @__ockl_printf_append_string_n(i64 [[BEGIN]], {{.*}}, i64 [[STRLEN1]], i32 0)
// CHECK: [[STRLEN2:%.*]] = phi i64 [ %{{[^,]*}}, %{{[^ ]*}} ], [ 0, %{{[^ ]*}} ]
// CHECK: [[APPEND2:%.*]] = call i64 @__ockl_printf_append_string_n(i64 [[APPEND1]], {{.*}}, i64 [[STRLEN2]], i32 0)
// CHECK: [[PTR2INT:%.*]] = ptrtoint ptr %{{.*}} to i64
// CHECK: [[APPEND3:%.*]] = call i64 @__ockl_printf_append_args(i64 [[APPEND2]], i32 1, i64 [[PTR2INT]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// CHECK: [[RETURN:%.*]]  = trunc i64 [[APPEND3]] to i32
// CHECK: ret i32 [[RETURN]]
