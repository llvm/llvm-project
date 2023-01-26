// RUN: %clang_cc1 -emit-llvm -triple arm64-- -o - %s -O0 | FileCheck %s --check-prefix=CHECK-O0
// RUN: %clang_cc1 -emit-llvm -disable-llvm-optzns -triple arm64-- -o - %s -O3 | FileCheck %s --check-prefix=CHECK-O3

struct large {
    void* pointers[8];
};

void pass_large(struct large);

// For arm64, we don't use byval to pass structs but instead we create
// temporary allocas.
//
// Make sure we generate the appropriate lifetime markers for the temporary
// allocas so that the optimizer can re-use stack slots if possible.
void example(void) {
    struct large l = {0};
    pass_large(l);
    pass_large(l);
}
// CHECK-O0-LABEL: define{{.*}} void @example(
// The alloca for the struct on the stack.
// CHECK-O0: %[[l:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
// The alloca for the temporary stack space that we use to pass the argument.
// CHECK-O0-NEXT: %[[byvaltemp:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
// Another one to pass the argument to the second function call.
// CHECK-O0-NEXT: %[[byvaltemp1:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
// First, memset `l` to 0.
// CHECK-O0-NEXT: call void @llvm.memset.p0.i64(ptr align 8 %[[l]], i8 0, i64 64, i1 false)
// Then, memcpy `l` to the temporary stack space.
// CHECK-O0-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[byvaltemp]], ptr align 8 %[[l]], i64 64, i1 false)
// Finally, call using a pointer to the temporary stack space.
// CHECK-O0-NEXT: call void @pass_large(ptr noundef %[[byvaltemp]])
// Now, do the same for the second call, using the second temporary alloca.
// CHECK-O0-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[byvaltemp1]], ptr align 8 %[[l]], i64 64, i1 false)
// CHECK-O0-NEXT: call void @pass_large(ptr noundef %[[byvaltemp1]])
// CHECK-O0-NEXT: ret void
//
// At O3, we should have lifetime markers to help the optimizer re-use the temporary allocas.
//
// CHECK-O3-LABEL: define{{.*}} void @example(
// The alloca for the struct on the stack.
// CHECK-O3: %[[l:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
// The alloca for the temporary stack space that we use to pass the argument.
// CHECK-O3-NEXT: %[[byvaltemp:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
// Another one to pass the argument to the second function call.
// CHECK-O3-NEXT: %[[byvaltemp1:[0-9A-Za-z-]+]] = alloca %struct.large, align 8
//
// Mark the start of the lifetime for `l`
// CHECK-O3-NEXT: call void @llvm.lifetime.start.p0(i64 64, ptr %[[l]])
//
// First, memset `l` to 0.
// CHECK-O3-NEXT: call void @llvm.memset.p0.i64(ptr align 8 %[[l]], i8 0, i64 64, i1 false)
//
// Lifetime of the first temporary starts here and ends right after the call.
// CHECK-O3-NEXT: call void @llvm.lifetime.start.p0(i64 64, ptr %[[byvaltemp]])
//
// Then, memcpy `l` to the temporary stack space.
// CHECK-O3-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[byvaltemp]], ptr align 8 %[[l]], i64 64, i1 false)
// Finally, call using a pointer to the temporary stack space.
// CHECK-O3-NEXT: call void @pass_large(ptr noundef %[[byvaltemp]])
//
// The lifetime of the temporary used to pass a pointer to the struct ends here.
// CHECK-O3-NEXT: call void @llvm.lifetime.end.p0(i64 64, ptr %[[byvaltemp]])
//
// Now, do the same for the second call, using the second temporary alloca.
// CHECK-O3-NEXT: call void @llvm.lifetime.start.p0(i64 64, ptr %[[byvaltemp1]])
// CHECK-O3-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[byvaltemp1]], ptr align 8 %[[l]], i64 64, i1 false)
// CHECK-O3-NEXT: call void @pass_large(ptr noundef %[[byvaltemp1]])
// CHECK-O3-NEXT: call void @llvm.lifetime.end.p0(i64 64, ptr %[[byvaltemp1]])
//
// Mark the end of the lifetime of `l`.
// CHECK-O3-NEXT: call void @llvm.lifetime.end.p0(i64 64, ptr %l)
// CHECK-O3-NEXT: ret void
