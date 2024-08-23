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
    this = Pair();
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
// CHECK:     define linkonce_odr noundef i32 @"?getFirst@Pair@@QAAHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %this) #0 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%Another = alloca %struct.Pair, align 4
// CHECK-NEXT:%agg.tmp = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:%First = getelementptr inbounds nuw %struct.Pair, ptr %Another, i32 0, i32 0
// CHECK-NEXT:store i32 5, ptr %First, align 4
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %Another, i32 0, i32 1
// CHECK-NEXT:store i32 10, ptr %Second, align 4
// CHECK-NEXT:call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %Another, i32 8, i1 false)
// CHECK-NEXT:call void @"??4Pair@@QAAXU0@@Z"(ptr noundef nonnull align 4 dereferenceable(8) %this1, ptr noundef byval(%struct.Pair) align 4 %agg.tmp)
// CHECK-NEXT:%First2 = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 0
// CHECK-NEXT:%0 = load i32, ptr %First2, align 4
// CHECK-NEXT:ret i32 %0

// CHECK:     define linkonce_odr noundef i32 @"?getSecond@Pair@@QAAHXZ"(ptr noundef nonnull align 4 dereferenceable(8) %this) #0 align 2 {
// CHECK-NEXT:entry:
// CHECK-NEXT:%this.addr = alloca ptr, align 4
// CHECK-NEXT:%agg.tmp = alloca %struct.Pair, align 4
// CHECK-NEXT:store ptr %this, ptr %this.addr, align 4
// CHECK-NEXT:%this1 = load ptr, ptr %this.addr, align 4
// CHECK-NEXT:call void @llvm.memset.p0.i32(ptr align 4 %agg.tmp, i8 0, i32 8, i1 false)
// CHECK-NEXT:call void @"??4Pair@@QAAXU0@@Z"(ptr noundef nonnull align 4 dereferenceable(8) %this1, ptr noundef byval(%struct.Pair) align 4 %agg.tmp)
// CHECK-NEXT:%Second = getelementptr inbounds nuw %struct.Pair, ptr %this1, i32 0, i32 1
// CHECK-NEXT:%0 = load i32, ptr %Second, align 4
// CHECK-NEXT:ret i32 %0
