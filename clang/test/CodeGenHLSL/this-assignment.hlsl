// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -emit-llvm -disable-llvm-passes -o - -hlsl-entry main %s | FileCheck %s

struct Pair {
  int First;
  int Second;

  int getFirst() {
    Pair Another = {5, 10};
    this = Another;
	  return this.First;
  }

  int getSecond() {
    this = Pair();
    return Second;
  }
};

[numthreads(1, 1, 1)]
void main() {
  Pair Vals = {1, 2.0};
  Vals.First = Vals.getFirst();
  Vals.Second = Vals.getSecond();
}

// This tests reference like implicit this in HLSL
// CHECK:     define linkonce_odr noundef i32 @"?getFirst@Pair@@QAAHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %this) #3 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%Another = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %Another, ptr align 4 @"__const.?getFirst@Pair@@QAAHXZ.Another", i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %this1, ptr align 4 %Another, i32 8, i1 false)
// CHECK-NEXT:%First = getelementptr inbounds %struct.Pair, ptr %this1, i32 0, i32 0

// CHECK:     define linkonce_odr noundef i32 @"?getSecond@Pair@@QAAHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %this) #3 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%ref.tmp = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memset.p0.i32(ptr align 4 %ref.tmp, i8 0, i32 8, i1 false)
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %this1, ptr align 4 %ref.tmp, i32 8, i1 false)
// CHECK-NEXT:%Second = getelementptr inbounds %struct.Pair, ptr %this1, i32 0, i32 1
