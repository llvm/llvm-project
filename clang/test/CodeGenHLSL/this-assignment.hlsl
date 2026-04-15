// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

struct Pair {
  int First;
  int Second;

  int getFirst() {
    Pair Another = {5, 10};
    this = Another;
    return this.First;
  }

  int getSecond() {
    this = {0, 0};
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
// CHECK-NEXT:%Another = alloca %struct.Pair, align 1
// CHECK-NEXT:%tmp = alloca %struct.Pair, align 1
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 1 %Another, ptr align 1 @__const._ZN4Pair8getFirstEv.Another, i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 1 %this1, ptr align 1 %Another, i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp, ptr align 1 %this1, i32 8, i1 false)
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 0
// CHECK-NEXT:%0 = load i32, ptr %First, align 1
// CHECK-NEXT:ret i32 %0

// CHECK-LABEL:     define {{.*}}getSecond
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%tmp = alloca %struct.Pair, align 1
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 0
// CHECK-NEXT:store i32 0, ptr %First, align 1
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 1
// CHECK-NEXT:store i32 0, ptr %Second, align 1
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp, ptr align 1 %this1, i32 8, i1 false)
// CHECK-NEXT:%Second2 = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 1
// CHECK-NEXT:%0 = load i32, ptr %Second2, align 1
// CHECK-NEXT:ret i32 %0

// CHECK-LABEL:     define {{.*}}DoSilly
// CHECK-NEXT: entry:
// CHECK-NEXT: [[ResultPtr:%.*]] = alloca ptr
// CHECK-NEXT: [[ThisPtrAddr:%.*]] = alloca ptr
// CHECK-NEXT: [[ObjIndirectAddr:%.*]] = alloca ptr
// CHECK-NEXT: [[Tmp:%.*]] = alloca %struct.Pair, align 1
// CHECK-NEXT: store ptr %agg.result, ptr [[ResultPtr]]
// CHECK-NEXT: store ptr {{.*}}, ptr [[ThisPtrAddr]]
// CHECK-NEXT: store ptr %Obj, ptr [[ObjIndirectAddr]]  
// CHECK-NEXT: [[ThisPtr:%.*]] = load ptr, ptr [[ThisPtrAddr]]
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[ThisPtr]], ptr align 1 %Obj, i32 8, i1 false)
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 1 [[Tmp]], ptr align 1 [[ThisPtr]], i32 8, i1 false)
// CHECK-NEXT: [[First4:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr [[ThisPtr]], i32 0, i32 0
// CHECK-NEXT: [[X:%.*]] = load i32, ptr [[First4]], align 1
// CHECK-NEXT: [[Add:%.*]] = add nsw i32 [[X]], 2
// CHECK-NEXT: store i32 [[Add]], ptr [[First4]], align 1
// CHECK-NEXT: [[First5:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr %agg.result, i32 0, i32 0
// CHECK-NEXT: [[First6:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr %Obj, i32 0, i32 0
// CHECK-NEXT: [[W:%.*]] = load i32, ptr [[First6]], align 1
// CHECK-NEXT: store i32 [[W]], ptr [[First5]], align 1
// CHECK-NEXT: [[Second7:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr %agg.result, i32 0, i32 1
// CHECK-NEXT: [[Second8:%.*]] = getelementptr inbounds nuw %struct.Pair, ptr %Obj, i32 0, i32 1
// CHECK-NEXT: [[V:%.*]] = load i32, ptr [[Second8]], align 1
// CHECK-NEXT: store i32 [[V]], ptr [[Second7]], align 1
// CHECK-NEXT: ret void
