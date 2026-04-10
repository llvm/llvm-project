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
    this = {0,0};
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
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%Another = alloca %struct.Pair, align 1
// CHECK-NEXT:%agg.tmp = alloca %struct.Pair, align 1
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 1 %Another, ptr align 1 @__const._ZN4Pair8getFirstEv.Another, i32 8, i1 false)
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %agg.tmp, i32 0, i32 0
// CHECK-NEXT:%First2 = getelementptr inbounds nuw %struct.Pair, ptr %Another, i32 0, i32 0
// CHECK-NEXT:%0 = load i32, ptr %First2, align 1
// CHECK-NEXT:store i32 %0, ptr %First, align 1
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %agg.tmp, i32 0, i32 1
// CHECK-NEXT:%Second3 = getelementptr inbounds nuw %struct.Pair, ptr %Another, i32 0, i32 1
// CHECK-NEXT:%1 = load i32, ptr %Second3, align 1
// CHECK-NEXT:store i32 %1, ptr %Second, align 1
// CHECK-NEXT:call void @_ZN4PairaSES_(ptr noundef nonnull align 1 dereferenceable(8) %this1, ptr noundef dead_on_return %agg.tmp)
// CHECK-NEXT:%First4 = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 0
// CHECK-NEXT:%2 = load i32, ptr %First4, align 1
// CHECK-NEXT:ret i32 %2

// CHECK:     define linkonce_odr hidden noundef i32 @_ZN4Pair9getSecondEv(ptr noundef nonnull align 1 dereferenceable(8) %this) #0 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%agg.tmp = alloca %struct.Pair, align 1
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %agg.tmp, i32 0, i32 0
// CHECK-NEXT:store i32 0, ptr %First, align 1
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %agg.tmp, i32 0, i32 1
// CHECK-NEXT:store i32 0, ptr %Second, align 1

// CHECK-NEXT:call void @_ZN4PairaSES_(ptr noundef nonnull align 1 dereferenceable(8) %this1, ptr noundef dead_on_return %agg.tmp)
// CHECK-NEXT:%Second2 = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 1
// CHECK-NEXT:%0 = load i32, ptr %Second2, align 1
// CHECK-NEXT:ret i32 %0
