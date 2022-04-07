// RUN:  %clang_cc1 -no-opaque-pointers -triple aarch64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o - %s | FileCheck --check-prefix IR %s

struct bt3 { signed b2:10; signed b3:10; } b16;

// Get the high 32-bits and then shift appropriately for big-endian.
signed callee_b0f(struct bt3 bp11) {
// IR: callee_b0f(i64 [[ARG:%.*]])
// IR: store i64 [[ARG]], i64* [[PTR:%.*]], align 8
// IR: [[BITCAST:%.*]] = bitcast i64* [[PTR]] to i8*
// IR: call void @llvm.memcpy.p0i8.p0i8.i64(i8* {{.*}}, i8* align 8 [[BITCAST]], i64 4
  return bp11.b2;
}
