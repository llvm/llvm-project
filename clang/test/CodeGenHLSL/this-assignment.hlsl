// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

struct Pair {
  int First;
  int Second;

  int getFirst() {
    Pair Another = {5, 10};
    this = Another;
	  return this.First;
  }

  // In HLSL 202x, this is a move assignment rather than a copy.
  int getSecond() {
    this = {0, 123};
    return Second;
  }

  // In HLSL 202x, this is a copy assignment.
  Pair DoSilly(Pair Obj) {
    this = Obj;
    First += 2;
    return Obj;
  }
};

[numthreads(1, 1, 1)]
void main() {
  Pair Vals = {1, 2.0};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();
  (void) Vals.DoSilly(Vals);
}

// This tests reference like implicit this in HLSL
// CHECK-LABEL:     define {{.*}}getFirst
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

// CHECK-LABEL:     define {{.*}}getSecond
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: [[ThisPtrAddr:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: store ptr %this, ptr [[ThisPtrAddr]], align 4
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAddr]], align 4
// CHECK-NEXT: [[First:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: store i32 0, ptr [[First]], align 1
// CHECK-NEXT: [[Second:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 1
// CHECK-NEXT: store i32 123, ptr [[Second]], align 1
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[ThisPtr]], i32 8, i1 false)
// CHECK-NEXT: [[Second2:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 1

// CHECK-LABEL:     define {{.*}}DoSilly
// CHECK-NEXT:entry:
// CHECK-NEXT: %[[#C_ENTRY:]] = call token @llvm.experimental.convergence.entry()
// CHECK-NEXT: [[ResultPtr:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[ThisPtrAddr:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[ObjIndirectAdds:%.*]] = alloca ptr, align 4
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: store ptr {{.*}}, ptr [[ResultPtr]]
// CHECK-NEXT: store ptr {{.*}}, ptr [[ThisPtrAddr]]
// CHECK-NEXT: store ptr {{.*}}, ptr [[ObjIndirectAdds]]
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAddr]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[ThisPtr]], ptr align 1 [[Obj:%.*]], i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[ThisPtr]], i32 8, i1 false)
// CHECK-NEXT: [[FirstAddr:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: [[First:%.*]] = load i32, ptr [[FirstAddr]]
// CHECK-NEXT: [[FirstPlusTwo:%.*]] = add nsw i32 [[First]], 2
// CHECK-NEXT: store i32 [[FirstPlusTwo]], ptr [[FirstAddr]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 {{.*}}, ptr align 1 [[Obj]], i32 8, i1 false)
