// RUN:  %clang_cc1 -triple aarch64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o %t -fdump-record-layouts-simple %s | FileCheck %s --check-prefix=LAYOUT
// RUN: FileCheck %s --check-prefix=IR <%t

struct bt3 { signed b2:10; signed b3:10; } b16;

// Get the high 32-bits and then shift appropriately for big-endian.
signed callee_b0f(struct bt3 bp11) {
// IR: callee_b0f(i64 [[ARG:%.*]])
// IR: [[BP11:%.*]] = alloca %struct.bt3, align 4
// IR: [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.bt3, ptr [[BP11]], i32 0, i32 0
// IR: [[COERCE_HIGHBITS:%.*]] = lshr i64 [[ARG]], 32
// IR: [[COERCE_VAL_II:%.*]] = trunc i64 [[COERCE_HIGHBITS]] to i32
// IR: store i32 [[COERCE_VAL_II]], ptr [[COERCE_DIVE]], align 4
// IR: [[BF_LOAD:%.*]] = load i32, ptr [[BP11]], align 4
// IR: [[BF_ASHR:%.*]] = ashr i32 [[BF_LOAD]], 22
// IR: ret i32 [[BF_ASHR]]
  return bp11.b2;
}

// LAYOUT-LABEL: LLVMType:%struct.bt3 =
// LAYOUT-SAME: type { i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:22 Size:10 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:12 Size:10 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>
