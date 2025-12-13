// RUN: %clang_cc1 -triple arm64e-apple-ios18 -fptrauth-calls -fptrauth-intrinsics -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_S0:.*]] = type { i32, i32, ptr }
// CHECK: %[[STRUCT_S1:.*]] = type { ptr, ptr }

// This struct isn't POD because it has an address-discriminated ptrauth
// field.
typedef struct {
  int f0, f1;
  int * __ptrauth(1,1,50) f2;
} S0;

// This struct isn't POD because it has an address-discriminated ptrauth
// field and an ARC ObjC pointer field.
typedef struct {
  id f0;
  int * __ptrauth(1,1,50) f1;
} S1;

// CHECK: define void @compound_literal_assignment0(ptr noundef %[[P:.*]])
// CHECK: %[[P_ADDR:.*]] = alloca ptr, align 8
// CHECK-NEXT: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_S0]], align 8
// CHECK-NEXT: store ptr %[[P]], ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[V0:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[F0:.*]] = getelementptr inbounds nuw %[[STRUCT_S0]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK-NEXT: %[[V1:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_S0]], ptr %[[V1]], i32 0, i32 1
// CHECK-NEXT: %[[V2:.*]] = load i32, ptr %[[F1]], align 4
// CHECK-NEXT: store i32 %[[V2]], ptr %[[F0]], align 8
// CHECK-NEXT: %[[F11:.*]] = getelementptr inbounds nuw %[[STRUCT_S0]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 1
// CHECK-NEXT: %[[V3:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[F02:.*]] = getelementptr inbounds nuw %[[STRUCT_S0]], ptr %[[V3]], i32 0, i32 0
// CHECK-NEXT: %[[V4:.*]] = load i32, ptr %[[F02]], align 8
// CHECK-NEXT: store i32 %[[V4]], ptr %[[F11]], align 4
// CHECK-NEXT: %[[F2:.*]] = getelementptr inbounds nuw %[[STRUCT_S0]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 2
// CHECK-NEXT: store ptr null, ptr %[[F2]], align 8
// CHECK-NEXT: call void @__copy_assignment_8_8_t0w8_pa1_50_8(ptr %[[V0]], ptr %[[_COMPOUNDLITERAL]])
// CHECK-NEXT: ret void

void compound_literal_assignment0(S0 *p) {
  *p = (S0){.f0 = p->f1, .f1 = p->f0};
}

// CHECK: define void @compound_literal_assignment1(ptr noundef %[[P:.*]])
// CHECK: %[[P_ADDR:.*]] = alloca ptr, align 8
// CHECK-NEXT: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_S1]], align 8
// CHECK-NEXT: store ptr %[[P]], ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[V0:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// CHECK-NEXT: %[[F0:.*]] = getelementptr inbounds nuw %[[STRUCT_S1]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK-NEXT: store ptr null, ptr %[[F0]], align 8
// CHECK-NEXT: %[[F1:.*]] = getelementptr inbounds nuw %[[STRUCT_S1]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 1
// CHECK-NEXT: store ptr null, ptr %[[F1]], align 8
// CHECK-NEXT: call void @__copy_assignment_8_8_s0_pa1_50_8(ptr %[[V0]], ptr %[[_COMPOUNDLITERAL]])
// CHECK-NEXT: call void @__destructor_8_s0(ptr %[[_COMPOUNDLITERAL]])
// CHECK-NEXT: ret void

void compound_literal_assignment1(S1 *p) {
  *p = (S1){};
}
