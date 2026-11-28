// Under -fdefined-int-divrem (the OpenCL default), the driver leaves
// integer-divide-by-zero and signed-integer-overflow-divide out of
// -fsanitize=undefined. When they are enabled explicitly, they are emitted as
// always, with their configured recover and trap behavior, ahead of the
// defined division.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s -fsanitize=integer-divide-by-zero,signed-integer-overflow-divide | FileCheck %s --check-prefixes=CHECK,FATAL
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s -fsanitize=integer-divide-by-zero,signed-integer-overflow-divide -fsanitize-recover=integer-divide-by-zero,signed-integer-overflow-divide | FileCheck %s --check-prefixes=CHECK,RECOVER
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s -fsanitize=integer-divide-by-zero,signed-integer-overflow-divide -fsanitize-trap=integer-divide-by-zero,signed-integer-overflow-divide | FileCheck %s --check-prefixes=CHECK,TRAP

// CHECK-LABEL: @f_sdiv(
// CHECK:       [[NZ:%.*]] = icmp ne i32 [[B:%.*]], 0
// CHECK:       [[NMIN:%.*]] = icmp ne i32 [[A:%.*]], -2147483648
// CHECK:       [[NM1:%.*]] = icmp ne i32 [[B]], -1
// CHECK:       [[OR:%.*]] = or i1 [[NMIN]], [[NM1]]
// CHECK:       and i1 [[NZ]], [[OR]]
// FATAL:       call void @__ubsan_handle_divrem_overflow_abort(
// FATAL-NEXT:  unreachable
// RECOVER:     call void @__ubsan_handle_divrem_overflow(
// TRAP:        call void @llvm.ubsantrap(i8 3)
// TRAP-NEXT:   unreachable
// CHECK:       [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
// CHECK-NEXT:  freeze i32 [[Q]]
int f_sdiv(int a, int b) { return a / b; }

// CHECK-LABEL: @f_urem(
// FATAL:       call void @__ubsan_handle_divrem_overflow_abort(
// RECOVER:     call void @__ubsan_handle_divrem_overflow(
// TRAP:        call void @llvm.ubsantrap(i8 3)
// CHECK:       [[Q:%.*]] = call i32 @llvm.masked.urem.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
// CHECK-NEXT:  freeze i32 [[Q]]
unsigned f_urem(unsigned a, unsigned b) { return a % b; }
