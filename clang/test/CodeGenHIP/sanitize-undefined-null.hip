// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -disable-llvm-passes -fcuda-is-device -fsanitize=null \
// RUN:   -o - %s | FileCheck --enable-var-scope %s

// Check there are no assertions when trying to sanitize when globals have non-0
// address spaces.

#define __device__ __attribute__((device))

//.
// CHECK: @.src = private unnamed_addr addrspace(4) constant [{{[0-9]+}} x i8] c
// CHECK: @0 = private unnamed_addr addrspace(1) constant { i16, i16, [7 x i8] } { i16 0, i16 7, [7 x i8] c"'char'\00" }
// CHECK: @1 = private unnamed_addr addrspace(1) global { { ptr, i32, i32 }, ptr addrspace(1), i8, i8 } { { ptr, i32, i32 } { ptr addrspacecast (ptr addrspace(4) @.src to ptr), i32 {{[0-9]+}}, i32 3 }, ptr addrspace(1) @0, i8 1, i8 1 }
//.
// CHECK-LABEL: @_Z3fooPc(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4, addrspace(5)
// CHECK-NEXT:    [[P_ADDR:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK-NEXT:    [[RETVAL_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[RETVAL]] to ptr
// CHECK-NEXT:    [[P_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[P_ADDR]] to ptr
// CHECK-NEXT:    store ptr [[P:%.*]], ptr [[P_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[P_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = icmp ne ptr [[TMP0]], null, !nosanitize !3
// CHECK-NEXT:    br i1 [[TMP1]], label [[CONT:%.*]], label [[HANDLER_TYPE_MISMATCH:%.*]], !prof [[PROF4:![0-9]+]], !nosanitize !3
// CHECK:       handler.type_mismatch:
// CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP0]] to i64, !nosanitize !3
// CHECK-NEXT:    call void @__ubsan_handle_type_mismatch_v1_abort(ptr addrspace(1) @[[GLOB1:[0-9]+]], i64 [[TMP2]]) #[[ATTR2:[0-9]+]], !nosanitize !3
// CHECK-NEXT:    unreachable, !nosanitize !3
// CHECK:       cont:
// CHECK-NEXT:    store i8 0, ptr [[TMP0]], align 1
// CHECK-NEXT:    ret i32 3
//
__device__ int foo(char *p) {
  *p = 0;
  return 3;
}
