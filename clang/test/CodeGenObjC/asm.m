// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -fobjc-arc -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_A:.*]] = type { ptr }

typedef struct {
  id f;
} A;

id a;

// Check that the compound literal is destructed at the end of the enclosing scope.

// CHECK-LABEL: define void @foo0()
// CHECK: %[[_COMPOUNDLITERAL:.*]] = alloca %[[STRUCT_A]], align 8
// CHECK: getelementptr inbounds %[[STRUCT_A]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK: %[[F1:.*]] = getelementptr inbounds %[[STRUCT_A]], ptr %[[_COMPOUNDLITERAL]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[F1]], align 8
// CHECK: call void asm sideeffect "", "r,~{dirflag},~{fpsr},~{flags}"(ptr %[[V2]])
// CHECK: call void asm sideeffect "",
// CHECK: call void @__destructor_8_s0(ptr %[[_COMPOUNDLITERAL]])

void foo0() {
  asm("" : : "r"(((A){a}).f) );
  asm("");
}
