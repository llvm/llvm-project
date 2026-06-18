// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - -std=hlsl202x %s | FileCheck %s

struct Pair {
  int First;
  int Second;
  int getFirst() {
    Pair Another = {5, 10};
    this = Another;
    return this.First;
  }
  int getSecond() {
    this = {0, 123};
    return Second;
  }
  void operator=(Pair P) {
    First = P.First;
    Second = 2;
  }
};
[numthreads(1, 1, 1)]
void main() {
  Pair Vals = {1, 2};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();
}

// This test makes a probably safe assumption that HLSL 202x includes operator overloading for assignment operators.
// CHECK:     define linkonce_odr hidden noundef i32 @_ZN4Pair8getFirstEv(ptr noundef nonnull align 1 dereferenceable(8) %this) #0 align 2 {
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: [[ThisPtrAdds:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[Another:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: store ptr %this, ptr [[ThisPtrAdds]], align 4
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAdds]], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Another]], ptr align 1 @__const._ZN4Pair8getFirstEv.Another, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[ThisPtr]], ptr align 1 [[Another]], i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[ThisPtr]], i32 8, i1 false)
// CHECK-NEXT: [[First:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: %[[LOAD:.*]] = load i32, ptr [[First]], align 1
// CHECK-NEXT: ret i32 %[[LOAD]]

// CHECK:     define linkonce_odr hidden noundef i32 @_ZN4Pair9getSecondEv(ptr noundef nonnull align 1 dereferenceable(8) %this) #0 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: [[ThisPtrAdds:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: store ptr %this, ptr [[ThisPtrAdds]], align 4
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAdds]], align 4
// CHECK-NEXT: [[First:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: store i32 0, ptr [[First]], align 1
// CHECK-NEXT: [[Second:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 1
// CHECK-NEXT: store i32 123, ptr [[Second]], align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[ThisPtr]], i32 8, i1 false)
// CHECK-NEXT: [[Second2:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 1
// CHECK-NEXT: %[[LOAD:.*]] = load i32, ptr [[Second2]], align 1
// CHECK-NEXT: ret i32 %[[LOAD]]
