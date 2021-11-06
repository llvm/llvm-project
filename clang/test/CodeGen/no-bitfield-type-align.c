// RUN: %clang_cc1 -triple x86_64-apple-darwin -fno-bitfield-type-align -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_S:.*]] = type { i32 }

struct S {
  unsigned short:   0;
  unsigned short  f1:15;
  unsigned short:   0;
  unsigned short  f2:15;
};

// CHECK: define{{.*}} void @test_zero_width_bitfield(%[[STRUCT_S]]* noundef %[[A:.*]])
// CHECK: %[[BF_LOAD:.*]] = load i32, i32* %[[V1:.*]], align 1
// CHECK: %[[BF_CLEAR:.*]] = and i32 %[[BF_LOAD]], 32767
// CHECK: %[[BF_CAST:.*]] = trunc i32 %[[BF_CLEAR]] to i16
// CHECK: %[[CONV:.*]] = zext i16 %[[BF_CAST]] to i32
// CHECK: %[[ADD:.*]] = add nsw i32 %[[CONV]], 1
// CHECK: %[[CONV1:.*]] = trunc i32 %[[ADD]] to i16
// CHECK: %[[V2:.*]] = zext i16 %[[CONV1]] to i32
// CHECK: %[[BF_LOAD2:.*]] = load i32, i32* %[[V1]], align 1
// CHECK: %[[BF_VALUE:.*]] = and i32 %[[V2]], 32767
// CHECK: %[[BF_CLEAR3:.*]] = and i32 %[[BF_LOAD2]], -32768
// CHECK: %[[BF_SET:.*]] = or i32 %[[BF_CLEAR3]], %[[BF_VALUE]]
// CHECK: store i32 %[[BF_SET]], i32* %[[V1]], align 1

// CHECK: %[[BF_LOAD4:.*]] = load i32, i32* %[[V4:.*]], align 1
// CHECK: %[[BF_LSHR:.*]] = lshr i32 %[[BF_LOAD4]], 15
// CHECK: %[[BF_CLEAR5:.*]] = and i32 %[[BF_LSHR]], 32767
// CHECK: %[[BF_CAST6:.*]] = trunc i32 %[[BF_CLEAR5]] to i16
// CHECK: %[[CONV7:.*]] = zext i16 %[[BF_CAST6]] to i32
// CHECK: %[[ADD8:.*]] = add nsw i32 %[[CONV7]], 2
// CHECK: %[[CONV9:.*]] = trunc i32 %[[ADD8]] to i16
// CHECK: %[[V5:.*]] = zext i16 %[[CONV9]] to i32
// CHECK: %[[BF_LOAD10:.*]] = load i32, i32* %[[V4]], align 1
// CHECK: %[[BF_VALUE11:.*]] = and i32 %[[V5]], 32767
// CHECK: %[[BF_SHL:.*]] = shl i32 %[[BF_VALUE11]], 15
// CHECK: %[[BF_CLEAR12:.*]] = and i32 %[[BF_LOAD10]], -1073709057
// CHECK: %[[BF_SET13:.*]] = or i32 %[[BF_CLEAR12]], %[[BF_SHL]]
// CHECK: store i32 %[[BF_SET13]], i32* %[[V4]], align 1

void test_zero_width_bitfield(struct S *a) {
  a->f1 += 1;
  a->f2 += 2;
}
