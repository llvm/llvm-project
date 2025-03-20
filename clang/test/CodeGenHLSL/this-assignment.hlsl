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
    this = Pair();
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
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%Another = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %Another, ptr align 4 @__const._ZN4Pair8getFirstEv.Another, i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %this1, ptr align 4 %Another, i32 8, i1 false)
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 0

// CHECK-LABEL:     define {{.*}}getSecond
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%ref.tmp = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memset.p0.i32(ptr align 4 %ref.tmp, i8 0, i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %this1, ptr align 4 %ref.tmp, i32 8, i1 false)
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 1

// CHECK-LABEL:     define {{.*}}DoSilly
// CHECK-NEXT:entry:
// CHECK-NEXT: [[ThisPtrAddr:%.*]] = alloca ptr
// CHECK-NEXT: store ptr {{.*}}, ptr [[ThisPtrAddr]]
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAddr]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[ThisPtr]], ptr align 4 [[Obj:%.*]], i32 8, i1 false)
// CHECK-NEXT: [[FirstAddr:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: [[First:%.*]] = load i32, ptr [[FirstAddr]]
// CHECK-NEXT: [[FirstPlusTwo:%.*]] = add nsw i32 [[First]], 2
// CHECK-NEXT: store i32 [[FirstPlusTwo]], ptr [[FirstAddr]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 {{.*}}, ptr align 4 [[Obj]], i32 8, i1 false)
