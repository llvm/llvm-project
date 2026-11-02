// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-HOSTCALL %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-BUFFERED %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcxx-exceptions -fexceptions -fclangir -fcuda-is-device -emit-llvm %s -o - | FileCheck --check-prefixes=LLVM,LLVM-CIR %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=LLVM,LLVM-OGCG %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BUFFERED,BUFFERED-CIR %s
// RUN: %clang_cc1 -triple=amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BUFFERED,BUFFERED-OGCG %s

// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-HOSTCALL %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-cir %s -o - \
// RUN: | FileCheck --check-prefixes=CIR,CIR-BUFFERED %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=LLVM,LLVM-CIR %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=LLVM,LLVM-OGCG %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BUFFERED,BUFFERED-CIR %s
// RUN: %clang_cc1 -triple=spirv64-amd-amdhsa -x hip -fcuda-is-device -fcxx-exceptions -fexceptions -mprintf-kind=buffered -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BUFFERED,BUFFERED-OGCG %s

// Device printf is emitted as cir.offload.printf, which AMDGPU expands into the
// OpenCL printf runtime: hostcall by default, or the buffered kind with
// -mprintf-kind=buffered.

#define __device__ __attribute__((device))
extern "C" __device__ int printf(const char *, ...);

// CIR-HOSTCALL: module {{.*}} attributes {{.*}}cir.amdgpu_printf_kind = "hostcall"
// CIR-BUFFERED: module {{.*}} attributes {{.*}}cir.amdgpu_printf_kind = "buffered"

// CIR-LABEL: cir.func {{.*}} @_Z1pi(
// CIR: %[[FMT:.*]] = cir.cast array_to_ptrdecay
// CIR: %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!s32i{{(, target_address_space\(4\))?}}>, !s32i
// CIR: cir.offload.printf(%[[FMT]], %[[V]]) : (!cir.ptr<!s8i{{(, target_address_space\(4\))?}}>, !s32i) -> !s32i
// LLVM-LABEL: @_Z1pi(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}@.str{{( to ptr.*\))?}}, i64 {{%.*}}, i32 0)
// LLVM: [[A:%.*]] = zext i32 {{%.*}} to i64
// LLVM: call{{.*}} i64 @__ockl_printf_append_args(i64 [[D1]], i32 1, i64 [[A]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// BUFFERED-LABEL: @_Z1pi(
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 20)
// BUFFERED: [[OK:%.*]] = icmp ne ptr addrspace(1) [[BUF]], null
// BUFFERED: br i1 [[OK]],
// BUFFERED: store i32 82, ptr addrspace(1) [[BUF]], align 4
// BUFFERED: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED: store i64 -3402830678189845200, ptr addrspace(1) [[P0]], align 8
// BUFFERED: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i32 8
// BUFFERED: [[A:%.*]] = zext i32 {{%.*}} to i64
// BUFFERED: store i64 [[A]], ptr addrspace(1) [[P1]], align 8
__device__ void p(int v) { printf("%d\n", v); }

// The result of printf must be forwarded to its users. The expansion's block
// layout differs between CIR and classic codegen, so the use is matched in any
// order.
// CIR-LABEL: cir.func {{.*}} @_Z1ri(
// CIR: %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR: %[[RES:.*]] = cir.offload.printf({{.*}})
// CIR: cir.store %[[RES]], %[[RETVAL]]
// LLVM-LABEL: @_Z1ri(
// LLVM-DAG: [[RES:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64
// LLVM-DAG: [[RET:%.*]] = trunc i64 [[RES]] to i32
// LLVM-DAG: {{store|ret}} i32 [[RET]]
// BUFFERED-LABEL: @_Z1ri(
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 20)
// BUFFERED: [[OK:%.*]] = icmp ne ptr addrspace(1) [[BUF]], null
// BUFFERED-DAG: [[FAIL:%.*]] = xor i1 [[OK]], true
// BUFFERED-DAG: [[RES:%.*]] = sext i1 [[FAIL]] to i32
// BUFFERED-DAG: {{store|ret}} i32 [[RES]]
__device__ int r(int v) { return printf("%d\n", v); }

// Device printf cannot throw, so a pending cleanup must not turn it into an
// invoke with an exception-handling path.
struct D { __device__ ~D(); };
// CIR-LABEL: cir.func {{.*}} @_Z2ehi(
// CIR: cir.cleanup.scope {
// CIR: cir.offload.printf({{.*}})
// CIR: } cleanup all {
// LLVM-LABEL: @_Z2ehi(
// LLVM-NOT: personality
// LLVM: call{{.*}} i64 @__ockl_printf_begin(i64
// LLVM-NOT: {{invoke|landingpad|resume}}
// BUFFERED-LABEL: @_Z2ehi(
// BUFFERED-NOT: personality
// BUFFERED: call{{.*}} ptr addrspace(1) @__printf_alloc(i32
// BUFFERED-NOT: {{invoke|landingpad|resume}}
__device__ void eh(int v) { D d; printf("%d\n", v); }

// %s arguments. A constant string is known at compile time; the length of a
// runtime string is computed by a loop. The buffered kind stores a constant
// string inline and copies a runtime string.
// CIR-LABEL: cir.func {{.*}} @_Z1sPKc(
// CIR: cir.offload.printf({{.*}}) : (!cir.ptr<!s8i{{(, target_address_space\(4\))?}}>, !cir.ptr<!s8i{{(, target_address_space\(4\))?}}>, !cir.ptr<!s8i{{.*}}>) -> !s32i
// LLVM-LABEL: @_Z1sPKc(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}@.str.1{{( to ptr.*\))?}}, i64 {{%.*}}, i32 0)
// LLVM: [[D2:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D1]], ptr{{.*}}@.str.2{{( to ptr.*\))?}}, i64 {{%.*}}, i32 0)
// LLVM: [[ISNULL:%.*]] = icmp eq ptr{{( addrspace\(4\))?}} [[STR:%[^,]+]], null
// LLVM: br i1 [[ISNULL]], label %[[JOIN:.*]], label %[[WHILE:.*]]
// LLVM: [[WHILE]]:
// LLVM: [[CUR:%.*]] = phi ptr
// LLVM: getelementptr i8, ptr{{( addrspace\(4\))?}} [[CUR]], i64 1
// LLVM: [[CH:%.*]] = load i8, ptr{{( addrspace\(4\))?}} [[CUR]], align 1
// LLVM: [[ISNUL:%.*]] = icmp eq i8 [[CH]], 0
// LLVM: br i1 [[ISNUL]], label %[[DONE:.*]], label %[[WHILE]]
// LLVM: [[DONE]]:
// LLVM: [[END:%.*]] = ptrtoaddr ptr{{( addrspace\(4\))?}} [[CUR]] to i64
// LLVM: [[BEGIN:%.*]] = ptrtoaddr ptr{{( addrspace\(4\))?}} [[STR]] to i64
// LLVM: [[DIFF:%.*]] = sub i64 [[END]], [[BEGIN]]
// LLVM: [[LEN:%.*]] = add i64 [[DIFF]], 1
// LLVM: br label %[[JOIN]]
// LLVM: [[JOIN]]:
// LLVM: [[SIZE:%.*]] = phi i64 [ [[LEN]], %[[DONE]] ], [ 0, %{{.*}} ]
// LLVM: call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D2]], ptr{{.*}}, i64 [[SIZE]], i32 1)
// BUFFERED-LABEL: @_Z1sPKc(
// BUFFERED: icmp eq ptr{{.*}}, null
// BUFFERED: [[LEN:%.*]] = phi i64
// BUFFERED: [[ADD:%.*]] = add i64 [[LEN]], 7
// BUFFERED: [[ALIGNED:%.*]] = and i64 [[ADD]], 4294967288
// BUFFERED: [[TOTAL:%.*]] = add i64 [[ALIGNED]], 20
// BUFFERED: [[SIZE:%.*]] = trunc i64 [[TOTAL]] to i32
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 [[SIZE]])
// BUFFERED: [[OK:%.*]] = icmp ne ptr addrspace(1) [[BUF]], null
// BUFFERED: br i1 [[OK]],
// BUFFERED: [[SHL:%.*]] = shl i32 [[SIZE]], 2
// BUFFERED: [[CTRL:%.*]] = or i32 [[SHL]], 2
// BUFFERED: store i32 [[CTRL]], ptr addrspace(1) [[BUF]], align 4
// BUFFERED: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED: store i64 5333508983605851890, ptr addrspace(1) [[P0]], align 8
// BUFFERED: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i32 8
// "cons" and "t\0" as little-endian i32 chunks.
// BUFFERED: store i32 1936617315, ptr addrspace(1) [[P1]], align 4
// BUFFERED: [[P2:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P1]], i32 4
// BUFFERED: store i32 116, ptr addrspace(1) [[P2]], align 4
// BUFFERED: [[P3:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P2]], i32 4
// BUFFERED: call{{.*}} void @llvm.memcpy.p1.{{p0|p4}}.i64(ptr addrspace(1){{.*}} [[P3]], ptr{{.*}}, i64 [[LEN]], i1 false)
__device__ void s(const char *str) { printf("%s|%s\n", "const", str); }

// A runtime format string. The buffered kind copies it instead of storing its
// hash, and does not set the constant-format bit in the control dword.
// CIR-LABEL: cir.func {{.*}} @_Z2nfPKci(
// CIR: cir.offload.printf({{.*}}) : (!cir.ptr<!s8i{{.*}}>, !s32i) -> !s32i
// LLVM-LABEL: @_Z2nfPKci(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: icmp eq ptr{{.*}}, null
// LLVM: [[LEN:%.*]] = phi i64
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}, i64 [[LEN]], i32 0)
// LLVM: [[A:%.*]] = zext i32 {{%.*}} to i64
// LLVM: call{{.*}} i64 @__ockl_printf_append_args(i64 [[D1]], i32 1, i64 [[A]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// BUFFERED-LABEL: @_Z2nfPKci(
// BUFFERED: [[LEN:%.*]] = phi i64
// BUFFERED: [[ADD:%.*]] = add i64 [[LEN]], 7
// BUFFERED: [[ALIGNED:%.*]] = and i64 [[ADD]], 4294967288
// BUFFERED: [[TOTAL:%.*]] = add i64 [[ALIGNED]], 12
// BUFFERED: [[SIZE:%.*]] = trunc i64 [[TOTAL]] to i32
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 [[SIZE]])
// BUFFERED: [[CTRL:%.*]] = shl i32 [[SIZE]], 2
// BUFFERED-NEXT: store i32 [[CTRL]], ptr addrspace(1) [[BUF]], align 4
// BUFFERED-NEXT: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED-NEXT: call{{.*}} void @llvm.memcpy.p1.{{p0|p4}}.i64(ptr addrspace(1){{.*}} [[P0]], ptr{{.*}}, i64 [[LEN]], i1 false)
// BUFFERED-NEXT: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i64 [[ALIGNED]]
// BUFFERED-NEXT: [[A:%.*]] = zext i32 {{%.*}} to i64
// BUFFERED-NEXT: store i64 [[A]], ptr addrspace(1) [[P1]], align 8
__device__ void nf(const char *fmt, int v) { printf(fmt, v); }

// No arguments: the format string is the last thing appended.
// CIR-LABEL: cir.func {{.*}} @_Z1zv(
// CIR: cir.offload.printf(%{{.*}}) : (!cir.ptr<!s8i{{(, target_address_space\(4\))?}}>) -> !s32i
// LLVM-LABEL: @_Z1zv(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}@.str.3{{( to ptr.*\))?}}, i64 {{%.*}}, i32 1)
// LLVM-NOT: @__ockl_printf_append_args
// LLVM: trunc i64 [[D1]] to i32
// BUFFERED-LABEL: @_Z1zv(
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 12)
// BUFFERED: store i32 50, ptr addrspace(1) [[BUF]], align 4
// BUFFERED: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED: store i64 3806265321777173681, ptr addrspace(1) [[P0]], align 8
__device__ void z() { printf("hello\n"); }

// Every promoted scalar kind. Each argument is appended separately and widened
// to 64 bits.
// CIR-LABEL: cir.func {{.*}} @_Z4manyildPvfcys(
// CIR: cir.offload.printf({{.*}}) : (!cir.ptr<!s8i{{(, target_address_space\(4\))?}}>, !s32i, !s64i, !cir.double, !cir.ptr<!void{{.*}}>, !cir.double, !s32i, !u64i, !s32i) -> !s32i
// LLVM-LABEL: @_Z4manyildPvfcys(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}@.str.4{{( to ptr.*\))?}}, i64 {{%.*}}, i32 0)
// LLVM: [[A1:%.*]] = zext i32 {{%.*}} to i64
// LLVM: [[D2:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D1]], i32 1, i64 [[A1]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[D3:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D2]], i32 1, i64 {{%.*}}, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[A3:%.*]] = bitcast double {{%.*}} to i64
// LLVM: [[D4:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D3]], i32 1, i64 [[A3]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[A4:%.*]] = ptrtoint ptr{{( addrspace\(4\))?}} {{%.*}} to i64
// LLVM: [[D5:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D4]], i32 1, i64 [[A4]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[A5:%.*]] = bitcast double {{%.*}} to i64
// LLVM: [[D6:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D5]], i32 1, i64 [[A5]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[A6:%.*]] = zext i32 {{%.*}} to i64
// LLVM: [[D7:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D6]], i32 1, i64 [[A6]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[D8:%.*]] = call{{.*}} i64 @__ockl_printf_append_args(i64 [[D7]], i32 1, i64 {{%.*}}, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 0)
// LLVM: [[A8:%.*]] = zext i32 {{%.*}} to i64
// LLVM: call{{.*}} i64 @__ockl_printf_append_args(i64 [[D8]], i32 1, i64 [[A8]], i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// BUFFERED-LABEL: @_Z4manyildPvfcys(
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 76)
// BUFFERED: store i32 306, ptr addrspace(1) [[BUF]], align 4
// BUFFERED: store i64 -3000127804524445155, ptr addrspace(1)
// BUFFERED: store i64 {{%.*}}, ptr addrspace(1)
// BUFFERED: store i64 {{%.*}}, ptr addrspace(1)
// BUFFERED: store double {{%.*}}, ptr addrspace(1)
// BUFFERED: store ptr{{( addrspace\(4\))?}} {{%.*}}, ptr addrspace(1)
// BUFFERED: store double {{%.*}}, ptr addrspace(1)
// BUFFERED: store i64 {{%.*}}, ptr addrspace(1)
// BUFFERED: store i64 {{%.*}}, ptr addrspace(1)
// BUFFERED: store i64 {{%.*}}, ptr addrspace(1)
__device__ void many(int a, long b, double c, void *d, float e, char f,
                     unsigned long long g, short h) {
  printf("%d %ld %f %p %f %c %llu %hd\n", a, b, c, d, e, f, g, h);
}

// A constant offset into a string literal still yields a constant string, so
// the format's %s is found and the argument is passed as a string, not as a
// pointer value.
// CIR-LABEL: cir.func {{.*}} @_Z3offv(
// CIR: %[[FMT:.*]] = cir.get_element
// CIR: %[[STR:.*]] = cir.ptr_stride
// CIR: cir.offload.printf(%[[FMT]], %[[STR]])
// LLVM-LABEL: @_Z3offv(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}} getelementptr {{.*}}@.str.5{{.*}}, i64 2), i64 {{%.*}}, i32 0)
// LLVM-NOT: @__ockl_printf_append_args
// LLVM: call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D1]], ptr{{.*}} getelementptr {{.*}}@.str.6{{.*}}, i64 5), i64 {{%.*}}, i32 1)
// BUFFERED-LABEL: @_Z3offv(
// BUFFERED: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 20)
// BUFFERED: store i32 82, ptr addrspace(1) [[BUF]], align 4
// BUFFERED: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED: store i64 748964781523690102, ptr addrspace(1) [[P0]], align 8
// BUFFERED: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i32 8
// "me\0" as a little-endian i32 chunk, then padding.
// BUFFERED: store i32 25965, ptr addrspace(1) [[P1]], align 4
// BUFFERED: [[P2:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P1]], i32 4
// BUFFERED: store i32 0, ptr addrspace(1) [[P2]], align 4
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstring-plus-int"
__device__ void off() { printf(&"xx%s|\n"[2], "skip-me" + 5); }
#pragma clang diagnostic pop

// A constant format in a deferred global. Classic codegen expands printf before
// the global's initializer is emitted, so it treats the format as non-constant
// and passes the %s argument as a pointer value.
// CIR-LABEL: cir.func {{.*}} @_Z2bfv(
// CIR: cir.get_global @_ZL9brace_fmt
// CIR: %[[FMT:.*]] = cir.cast array_to_ptrdecay
// CIR: %[[STR:.*]] = cir.cast array_to_ptrdecay
// CIR: cir.offload.printf(%[[FMT]], %[[STR]])
// LLVM-LABEL: @_Z2bfv(
// LLVM: [[D0:%.*]] = call{{.*}} i64 @__ockl_printf_begin(i64 0)
// LLVM: [[D1:%.*]] = call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D0]], ptr{{.*}}@_ZL9brace_fmt{{.*}}, i64 {{%.*}}, i32 0)
// LLVM-CIR-NOT: @__ockl_printf_append_args
// LLVM-CIR: call{{.*}} i64 @__ockl_printf_append_string_n(i64 [[D1]], ptr{{.*}}@.str.7{{.*}}, i64 {{%.*}}, i32 1)
// LLVM-OGCG: call{{.*}} i64 @__ockl_printf_append_args(i64 [[D1]], i32 1, i64 ptrtoint ({{.*}}@.str.7{{.*}}), i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i32 1)
// BUFFERED-LABEL: @_Z2bfv(
// BUFFERED-CIR: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 20)
// BUFFERED-CIR: store i32 82, ptr addrspace(1) [[BUF]], align 4
// BUFFERED-CIR: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED-CIR: store i64 -88937716022153584, ptr addrspace(1) [[P0]], align 8
// BUFFERED-CIR: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i32 8
// BUFFERED-CIR: store i32 120, ptr addrspace(1) [[P1]], align 4
// BUFFERED-CIR: [[P2:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P1]], i32 4
// BUFFERED-CIR: store i32 0, ptr addrspace(1) [[P2]], align 4
// BUFFERED-OGCG: [[LEN:%.*]] = phi i64
// BUFFERED-OGCG: [[ADD:%.*]] = add i64 [[LEN]], 7
// BUFFERED-OGCG: [[ALIGNED:%.*]] = and i64 [[ADD]], 4294967288
// BUFFERED-OGCG: [[TOTAL:%.*]] = add i64 [[ALIGNED]], 12
// BUFFERED-OGCG: [[SIZE:%.*]] = trunc i64 [[TOTAL]] to i32
// BUFFERED-OGCG: [[BUF:%.*]] = call{{.*}} ptr addrspace(1) @__printf_alloc(i32 [[SIZE]])
// BUFFERED-OGCG: [[CTRL:%.*]] = shl i32 [[SIZE]], 2
// BUFFERED-OGCG-NEXT: store i32 [[CTRL]], ptr addrspace(1) [[BUF]], align 4
// BUFFERED-OGCG-NEXT: [[P0:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[BUF]], i32 4
// BUFFERED-OGCG-NEXT: call{{.*}} void @llvm.memcpy.p1.{{p0|p4}}.i64(ptr addrspace(1){{.*}} [[P0]], ptr{{.*}}@_ZL9brace_fmt{{.*}}, i64 [[LEN]], i1 false)
// BUFFERED-OGCG-NEXT: [[P1:%.*]] = getelementptr inbounds i8, ptr addrspace(1) [[P0]], i64 [[ALIGNED]]
// BUFFERED-OGCG-NEXT: store ptr{{.*}}@.str.7{{.*}}, ptr addrspace(1) [[P1]], align 8
const char brace_fmt[] = {'%', 's', '\n', 0};
__device__ void bf() { printf(brace_fmt, "x"); }

// Constant formats are recorded with the low 64 bits of their MD5 hash, in
// emission order. The runtime format in nf adds no entry, nor does brace_fmt
// in classic codegen.
// LLVM-NOT: !llvm.printf.fmts
// BUFFERED-CIR: !llvm.printf.fmts = !{[[F0:![0-9]+]], [[F0]], [[F0]], [[F1:![0-9]+]], [[F2:![0-9]+]], [[F3:![0-9]+]], [[F4:![0-9]+]], [[F5:![0-9]+]]}
// BUFFERED-OGCG: !llvm.printf.fmts = !{[[F0:![0-9]+]], [[F0]], [[F0]], [[F1:![0-9]+]], [[F2:![0-9]+]], [[F3:![0-9]+]], [[F4:![0-9]+]]}
// BUFFERED-DAG: [[F0]] = !{!"0:0:d0c6b786f3b15530,%d\0A"}
// BUFFERED-DAG: [[F1]] = !{!"0:0:4a046e29962e76f2,%s|%s\0A"}
// BUFFERED-DAG: [[F2]] = !{!"0:0:34d29224c96a94b1,hello\0A"}
// BUFFERED-DAG: [[F3]] = !{!"0:0:d65d67a83a8c421d,%d %ld %f %p %f %c %llu %hd\0A"}
// BUFFERED-DAG: [[F4]] = !{!"0:0:a64db8027ff6276,%s|\0A"}
// BUFFERED-CIR-DAG: [[F5]] = !{!"0:0:fec4079e555d6e90,%s\0A"}
